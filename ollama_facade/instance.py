import subprocess


def check_ollama_instance_running():
    try:
        ollama_process = subprocess.run("ollama -v", capture_output=True, timeout=30)
        result = ollama_process.stdout.decode("utf-8")
        if "could not connect to a running Ollama instance" in result:
            raise Exception("ollama service must be running")
    except Exception as e:
        if e is subprocess.TimeoutExpired:
            raise Exception("ollama could not be found or started")
        raise Exception("an exception occured {}".format(e))
