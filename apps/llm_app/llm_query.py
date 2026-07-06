import fire


DEMO = (
    "Query an LLM model for an answer to a given prompt: "
    "{'app': 'llm', 'action': 'complete_text', 'prompt': [PROMPT]}"
)


def construct_action(
    word_dir, args: dict, py_file_path="/apps/llm_app/llm_query.py"
):
    import shlex
    return "python3 {} --prompt {}".format(py_file_path, shlex.quote(args["prompt"]))


def query(prompt):
    """
    Query an LLM model for an answer to a given prompt.
    """
    try:
        import os
        from openai import AzureOpenAI

        client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        )
        response = client.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=200,
            top_p=1,
            n=1
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"


def main(prompt):
    response = query(prompt)
    return response


if __name__ == "__main__":
    fire.Fire(main)
