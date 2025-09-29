# App that runs with FastAPI in MCP inside the docker


# Build the image and run the container
docker-compose up -d --build
docker-compose up -d --force-recreate --build


## Check the connecion to the LLM
--Connected to a gemma3 that is running in a docker container/  standalone_llm
import requests
import json
                            ############################ 
                            ##    Open AI endpoint    ##
                            ############################
url = "http://localhost:11434/v1/chat/completions"
payload = {"model": "gemma3", 
           "messages": [{"role": "user", "content": 'hi'}], "stream": False
            }
response = requests.post(url, json=payload, timeout=300)
result = response.json()
print(result['choices'][0]['message']['content'])


                            ############################ 
                            ## Native Ollama endpoint ##
                            ############################
model = "gemma3"
prompt = "hi"
url = "http://localhost:11434/api/generate"  
payload = { "model": model,
    "prompt": prompt,
    "stream": False
}
response = requests.post(url, json=payload, timeout=200)
result = response.json()
print(result['response'])  # Note: different response structure