import google.generativeai as genai
import PIL.Image
import dotenv
import os
# Load environment variables   

dotenv.load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=google_api_key)

gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")

# Read and create image object
image = PIL.Image.open('screenshots/screenshot_1747773094.jpeg')

# Generate content with image
response = gemini_model.generate_content(
    [
        image,
        'Caption this image.'
    ]
)

print(response.text)
