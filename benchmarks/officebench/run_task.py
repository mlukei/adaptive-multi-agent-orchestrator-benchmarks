"""Internal subprocess worker — executes a single benchmark task.
"""

import json
import os
import shutil

import fire
from dotenv import load_dotenv

from runtime.config import load_config
from runtime.experiment_runtime import make_run_tag
from benchmarks.officebench.evaluation import run_task_evaluation
from benchmarks.officebench.policy import OfficeBenchPolicy
from benchmarks.officebench.docker_build import build_docker
from benchmarks.officebench.environment import OfficeAgentEnv

# Load .env once at the entry-point level.
load_dotenv()

# Load central YAML config
_CFG = load_config()


def main(
    docker_name=None,
    container_name=None,
    dockerfile_path=None,
    model_name=None,
    task_dir='tasks/1-1',
    config_file='tasks/1-1/subtasks/0.json',
    task=None,
    tag=None,
    mode=None,
    task_index=0,
):
    # Apply YAML defaults
    docker_name = docker_name or _CFG.docker.image_name
    container_name = container_name or _CFG.docker.container_name
    dockerfile_path = dockerfile_path or _CFG.docker.dockerfile_path
    model_name = model_name or _CFG.llm.model_name
    mode = mode or _CFG.execution.mode

    # Build Docker image (if not exists)
    build_docker(docker_name, dockerfile_path)

    # Load task config
    with open(config_file, encoding="utf-8") as f:
        config = json.load(f)
    task = config.get('task', task)
    subtask_id = config_file.split('/')[-1].split('.')[0]

    # Prepend context metadata (date, user, etc.) so the orchestrator knows
    # the simulated current date and acting user.
    context_parts = []
    if 'username' in config:
        context_parts.append(f"Current user: {config['username']}")
    if 'date' in config:
        context_parts.append(f"Today's date: {config['date']}")
    if 'weekday' in config:
        context_parts.append(f"Day of week: {config['weekday']}")
    if 'time' in config:
        context_parts.append(f"Current time: {config['time']}")
    if context_parts:
        task = f"[Context: {', '.join(context_parts)}]\n\n{task}"
    config['task_dir'] = task_dir
    config['subtask_id'] = subtask_id

    # Generate unique tag if not provided
    if tag is None:
        tag = make_run_tag()

    # Output directory (with model_name for evaluation)
    model_name_normalized = model_name.replace('/', '_')
    output_dir = f'{task_dir}/outputs/{subtask_id}/{model_name_normalized}_{tag}'

    # Check if output already exists
    if os.path.exists(output_dir):
        if mode == 'force_new':
            print(f"Removing existing output: {output_dir}")
            shutil.rmtree(output_dir)
        else:
            print(f"Output already exists: {output_dir}")
            return

    # Start Docker environment
    env = OfficeAgentEnv(
        image_name=docker_name,
        container_name=container_name,
        task=task,
        verbose=True,
    )
    env.reset()
    env.prepare_docker_env(testbed_dir=f'{task_dir}/testbed/', app_dir='apps/')
    env.cache_docker_status(local_cache_dir=f'{task_dir}/cache/{subtask_id}/')

    # Create and run policy
    policy = OfficeBenchPolicy(
        model_name=model_name,
        env=env,
        task_config=config,
        tag=tag,
    )

    result = None
    run_exception: Exception | None = None
    interrupted = False
    try:
        result = policy.run()
    except KeyboardInterrupt:
        print("Interrupted.")
        interrupted = True
    except Exception as exc:
        run_exception = exc

    # Save results
    print(f"Saving outputs to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    env.cache_docker_status(local_cache_dir=output_dir)
    env.dump_history(output_dir)
    with open(f'{output_dir}/settings.json', 'w') as f:
        json.dump(config | {'model_name': model_name}, f, indent=2)

    # Evaluate against the persisted output testbed 
    eval_result = None
    if not interrupted and run_exception is None:
        try:
            eval_result = run_task_evaluation(
                task_dir,
                subtask_id,
                testbed_dir=os.path.join(output_dir, "testbed"),
            )
            if eval_result is None:
                print("No evaluation config for this task.")
            else:
                print(f"Task evaluation: {'PASSED' if eval_result else 'FAILED'}")
        except Exception as exc:
            print(f"Task evaluation crashed: {exc!r}")
            eval_result = False

    # Write one run summary row.
    policy.log_last_run_summary(eval_result=eval_result)

    print(f"Closing environment for container: {container_name}")
    env.close()
    print("Environment closed.")

    if run_exception is not None:
        raise run_exception
    return result


if __name__ == '__main__':
    fire.Fire(main)
