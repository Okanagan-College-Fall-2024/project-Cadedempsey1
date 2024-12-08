import requests
import json

class ChatGPTRequest:
    def __init__(self, temperature=0.3, model='gpt-3.5-turbo', api_key_path=None):
        """
        Initializes the ChatGPTRequest class.
        :param temperature: Controls the randomness of the response (0.0 to 1.0).
        :param model: The OpenAI model to use, e.g., 'gpt-3.5-turbo' or 'gpt-4'.
        :param api_key_path: Path to the file containing the OpenAI API key.
        """
        self.temperature = temperature
        self.model = model
        self.results = []
        self.api_key = self._read_api_key(api_key_path or "/Users/daemo/419/openAI.txt")

    def _read_api_key(self, api_key_path):
        """
        Reads the OpenAI API key from a file.what
        :param api_key_path: Path to the file containing the API key.
        :return: The API key as a string.
        """
        try:
            with open(api_key_path, "r") as file:
                return file.readline().strip()
        except FileNotFoundError:
            raise FileNotFoundError(f"API key file not found at {api_key_path}")
        except Exception as e:
            raise Exception(f"Error reading API key: {str(e)}")

    def send_request(self, prompt):
        """
        Sends a custom request to the OpenAI API.
        :param prompt: The prompt text to send to the API.
        :return: The response content or an error message.
        """
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }

        try:
            response = requests.post(url, headers=headers, json=data)
            print(f"Request status: {response.status_code}")
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}, {response.text}"
        except requests.exceptions.RequestException as e:
            return f"Request failed: {str(e)}"

# Example usage
if __name__ == "__main__":
    # Customize the parameters as needed
    chat_request = ChatGPTRequest(temperature=0.7, model="gpt-3.5-turbo")

    # Input a custom prompt
    prompt = input("Enter your prompt: ")

    # Send the request and get the response
    response = chat_request.send_request(prompt)

    # Display the response
    print("\nChatGPT Response:")
    print(response)