import json
import math
import os
import shlex
import subprocess
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer

from intercode.envs.ic_env import (
    IntercodeEnv,
    AGENT_OBS, EVAL_OBS, CORRUPT_GOLD, ACTION_EXEC, REWARD
)
from intercode.utils import get_container, timeout

import logging

logger = logging.getLogger(__name__)

# In the future, all of these functions should be imported from a single file
import apps


GIT_RESET_SCRIPT = "git reset --hard; git clean -fd;"
GIT_STATUS_SCRIPT = "git status --short;"

class OfficeAgentEnv(IntercodeEnv):
    """Gym environment for bash shell"""
    name = "officeagent_bash"

    def __init__(self, image_name: str, container_name: str, **kwargs):
        super(OfficeAgentEnv, self).__init__(image_name, container_name, **kwargs)
        # OfficeAgent states
        self.task = kwargs.get("task", '<undefined>')
        self.current_app = None
        self.available_apps = apps.AVAILABLE_APPS
        self.available_agents = apps.AVAILABLE_AGENTS_INTRO
        self.history = []

        self.logger = logging.getLogger(__name__)

    def prepare_docker_env(self, testbed_dir, app_dir):
        self._prepare_docker_testbed(testbed_dir)
        self._prepare_docker_apps(app_dir)

    def _prepare_docker_testbed(self, testbed_dir):
        os.makedirs(testbed_dir, exist_ok=True)
        for subdir in ("data", "emails", "calendar"):
            os.makedirs(os.path.join(testbed_dir, subdir), exist_ok=True)
        command = [
            'docker', 'cp', f'{testbed_dir}', f'{self.container_name}:/'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"Prepare Testbed: Failed to copy testbed to container: {result.stderr}" 
    
    def _prepare_docker_apps(self, app_dir):
        command = [
            'docker', 'cp', f'{app_dir}', f'{self.container_name}:/'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"Prepare Apps: Failed to copy apps to container: {result.stderr}"

    def cache_docker_status(self, local_cache_dir, remote_cache_dir="/testbed"):
        os.makedirs(local_cache_dir, exist_ok=True)
        command = [
            'docker', 'cp', f'{self.container_name}:{remote_cache_dir}', f'{local_cache_dir}/'
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(
                "Cache Status: Failed to copy cache from container %s (already stopped?): %s",
                self.container_name, result.stderr.strip(),
            )

    def dump_history(self, output_dir):
        with open(f"{output_dir}/env_history.json", "w") as f:
            json.dump(self.history, f, indent=2)
    
    def _write_answer_to_docker(self, answer: str, file_path: str):
        answer = str(answer)
        answer = answer.replace('"', '').replace("'", '')
        command = [
            'docker', 'exec', self.container_name, 'bash', '-c', "echo '{}' > {}".format(answer, file_path)
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        assert result.returncode == 0, f"Write Answer: Failed to write answer to container: {result.stderr}"


    def get_available_actions(self) -> list:
        current_app = self.current_app
        available_actions = list(apps.AVAILABLE_ACTIONS[current_app].keys())
        return available_actions

    def reset_container(self) -> None:
        self.workdir = "/"
        exit_code, output = self.container.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.ctr_name_eval}` container successfully: {output}")

    def check_valid_action(self, action: dict) -> bool:
        """Checks if action is valid"""
        try:
            assert 'app' in action and 'action' in action
            return True
        except AssertionError:
            return False
    
    def _minor_action_fix(self, action):
        proc_action = {}
        for k, v in action.items():
            if isinstance(v, list) and len(v) == 1:
                proc_action[k] = v[0]
            else:
                proc_action[k] = v
        return proc_action

    def exec_action(self, action_string: str) -> None:
        self.observation = None
        try:
            action = eval(action_string)
            action = self._minor_action_fix(action)
            assert self.check_valid_action(action)

            # special case for switch app
            if action['action'] == 'switch_app':
                action['app'] = 'system'

            is_cd_flag = False
            if action["app"] == "shell":
                command = action["command"]
                if isinstance(command, list):
                    command = ' '.join(command)
                # Only treat as a pure cd if the command is "cd <path>" with no
                # additional chained operators (&&, ;, |, ||).
                stripped = command.strip()
                is_cd_flag = stripped.startswith("cd") and not any(
                    op in stripped for op in ['&&', '||', ';', '|']
                )
                if is_cd_flag:
                    cd_arg = stripped[stripped.index("cd ")+3:].strip()
                    new_path = self.simplify_path(self.workdir, cd_arg)
                    command = f"cd {new_path}"
            elif action["app"] == "system":
                if action["action"] == "switch_app":
                    self.current_app = action["target_app"]
                    self.observation = f"Successfully switched to app: {self.current_app}"
                elif action["action"] == "finish_task":
                    answer = action.get("answer", 'None')
                    self._write_answer_to_docker(answer, "/testbed/data/answer.txt")
                    self.observation = "Task finished"
                elif action["action"] == 'got_stuck':
                    answer = 'None'
                    self.observation = "Task failed"
                command = None
            else:
                action_module = apps.AVAILABLE_ACTIONS[action["app"]][action["action"]]
                command = action_module.construct_action(self.workdir, args=action)

            if command is not None:
                with timeout():
                    cleaned_cmd = self.clean_cmd(command)
                    self.logger.info(f"Executing command: [{cleaned_cmd}]")
                    # Pass Azure OpenAI environment variables to container
                    env_vars = {
                        'AZURE_OPENAI_ENDPOINT': os.getenv('AZURE_OPENAI_ENDPOINT', ''),
                        'AZURE_OPENAI_API_KEY': os.getenv('AZURE_OPENAI_API_KEY', ''),
                        'AZURE_OPENAI_API_VERSION': os.getenv('AZURE_OPENAI_API_VERSION', ''),
                        'AZURE_OPENAI_DEPLOYMENT': os.getenv('AZURE_OPENAI_DEPLOYMENT', '')
                    }
                    
                    exit_code, output = self.container.exec_run(
                        cleaned_cmd,
                        workdir=self.workdir,
                        environment=env_vars
                    )
                    self.observation = output.decode("utf-8").split('OBSERVATION:')[-1].strip()
                    self.info[ACTION_EXEC] = exit_code == 0

                if is_cd_flag and self.info[ACTION_EXEC]:
                    self.workdir = new_path
                
                if self.observation == "" and action["app"] == "shell":
                    self.observation = f"Successfully executed command: {command}. The output was [{output.decode('utf-8')}]."

            self.history.append((action, self.observation))
        except (SyntaxError, ValueError, AssertionError) as e:
            self.logger.warning("exec_action – malformed action: %s", e)
            self.observation = "Malformed action! You must follow the given action format! Try a different action."
            self.info[ACTION_EXEC] = False
            self.history.append((action_string, self.observation))
        except Exception as e:
            self.logger.error("exec_action – execution error: %s: %s", type(e).__name__, e)
            self.observation = f"Action failed: {type(e).__name__}: {e}"
            self.info[ACTION_EXEC] = False
            self.history.append((action_string, self.observation))
        return
            
    def get_reward(self) -> tuple[float, dict[str, Any]]:
        """
        The reward currently is calculated as a weighted sum of the following:
        - 0.33: (File System Diff) Difference in file system states between agent, gold command
        - 0.33: (File Content) Verify each file was correctly changed by agent using hashing
        - 0.33: (Observation) Verify that correct output was generated
        """
        # Reset evaluation container state
        exit_code, output = self.container_eval.exec_run(self.clean_cmd(GIT_RESET_SCRIPT))
        if exit_code != 0:
            raise RuntimeError(f"Failed to reset `{self.ctr_name_eval}` container successfully: {output}")
        
        # Run gold command(s) in evaluation container
        self.observation_eval = None
        try:
            if isinstance(self.gold, str):
                self.observation_eval = self.container_eval.exec_run(
                    self.clean_cmd(self.gold)).output.decode("utf-8")
            elif isinstance(self.gold, list):
                self.observation_eval = self.container_eval.exec_run(
                self.clean_cmd(";".join(self.gold))).output.decode("utf-8")
            self.info[CORRUPT_GOLD] = False
        except Exception as e:
            self.info[CORRUPT_GOLD] = True

        # Calculate Rewards
        reward, info = 0.01, {}
        info[REWARD] = {}

        # PART 1: Compare file system states
        diff_agent = self.parse_status(self.container.exec_run(self.clean_cmd(GIT_STATUS_SCRIPT)).output.decode("utf-8"))
        diff_eval = self.parse_status(self.container_eval.exec_run(self.clean_cmd(GIT_STATUS_SCRIPT)).output.decode("utf-8"))
        info["diff_miss"] = list(set(diff_eval) - set(diff_agent))
        info["diff_extra"] = list(set(diff_agent) - set(diff_eval))
        p1_score = round(0.33 * (1 - math.erf(len(info["diff_miss"]) + len(info["diff_extra"]))), 2)
        info[REWARD]["file_diff"] = p1_score
        reward += p1_score

        # PART 2: Check if files changed by both agent, gold commands were modified correctly
        p2_score = 0.33
        # Only check corrects of common changes that were added or modified
        filter_changes = lambda x: (x[1] in ["A", "??", "C"])
        diff_same = [x for x in list(set(diff_agent) & set(diff_eval)) if filter_changes(x)]
        
        if len(diff_same) > 0:
            same_changes = 0
            # Compute hashes for files and folders differently using md5 checksums
            get_hash_cmd = lambda x: f"md5sum {x}" if "." in x else f"md5deep -r {x}"

            for path in diff_same:
                hash_cmd = get_hash_cmd(path[0])
                agent_hash = self.container.exec_run(hash_cmd).output.decode("utf-8")
                gold_hash = self.container_eval.exec_run(hash_cmd).output.decode("utf-8")
                same_changes += 1 if agent_hash == gold_hash else 0
            
            info["diff_same"] = {"files": diff_same, "correct": same_changes, "total": len(diff_same)}
            p2_score = round(0.33 * (same_changes / len(diff_same)), 2)
        info[REWARD]["file_changes"] = p2_score
        reward += p2_score
        
        # PART 3: Compare agent, query answers
        info[AGENT_OBS] = self.observation
        info[EVAL_OBS] = self.observation_eval
        try:
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([info[AGENT_OBS], info[EVAL_OBS]])
            answer_similarity = tfidf * tfidf.T
            info["answer_similarity"] = answer_similarity.toarray()[0][1]
        except:
            info["answer_similarity"] = 1 if info[AGENT_OBS] == info[EVAL_OBS] else 0
        p3_score = round(0.33 * info["answer_similarity"], 2)
        info[REWARD]["answer_similarity"] = p3_score
        reward += p3_score

        self.reward = reward 
        self.info.update(info)

        self.logger.info(f"Info: {self.info}")
        self.logger.info(f"Reward: {self.reward}")
        return reward, info

    def _cleanup_container(self, container_name: str, timeout_seconds: int = 15) -> None:
        """Best-effort container shutdown that cannot block indefinitely."""
        if not container_name:
            return

        stop_cmd = ["docker", "stop", "--time", "1", container_name]
        rm_cmd = ["docker", "rm", "-f", container_name]

        for cmd in (stop_cmd, rm_cmd):
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                )
            except subprocess.TimeoutExpired:
                self.logger.warning(
                    "Timed out while running `%s` for container `%s`",
                    " ".join(cmd[:2]),
                    container_name,
                )
                continue

            if result.returncode == 0:
                continue

            stderr = (result.stderr or "").strip()
            if "No such container" in stderr:
                return
            if stderr:
                self.logger.warning(
                    "Container cleanup command failed (`%s`): %s",
                    " ".join(cmd),
                    stderr,
                )

    def close(self):
        self.logger.info("Beginning environment shutdown...")
        self._cleanup_container(getattr(self, "container_name", ""))
        self._cleanup_container(getattr(self, "ctr_name_eval", ""))
        self.logger.info("Environment containers shutdown complete")
    
    ############################
    ### MARK: Helper methods ###
    ############################

    def clean_cmd(self, action: str) -> str:
        """Cleans action string"""
        entrypoint = "/bin/bash" # IMAGE_TO_SETTINGS[self.image_name]
        return f"{entrypoint} -c {shlex.quote(action.strip())}"

    def parse_status(self, status: str) -> list:
        """Parses git status output into list of changes"""
        status_lst = status.split()
        changes = []
        for i in range(0, len(status_lst), 2):
            changes.append((status_lst[i+1], status_lst[i]))
        return changes

    def simplify_path(self, current: str, changed: str) -> str:
        """Resolves path from current working directory path and the argument of the `cd` command"""
        if not changed:
            return current
        if changed[0] == "/":
            current = ""

        path = []
        
        for segment in (current + "/" + changed).split("/"):
            if segment == "..":
                if path:
                    path.pop()
            elif segment and segment != ".":
                path.append(segment)

        return "/" + "/".join(path)
