from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import easyocr
import cv2
import numpy as np
import re
from groq import Groq
import httpx
import ssl
import certifi

ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl._create_default_https_context = lambda: ssl_context

app = FastAPI()

ocr = easyocr.Reader(['ar', 'en'])
client = Groq(
    api_key="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",
    http_client=httpx.Client(verify=False)
)
model = "llama-3.3-70b-versatile"

def is_arabic(text):
    arabic_pattern = re.compile('[\u0600-\u06FF]')
    return bool(arabic_pattern.search(text))

def process_arabic_ocr(image_np):
    img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
    result = ocr.readtext(img_rgb)
    extracted_text = [text for (_, text, _) in result]
    return extracted_text

def generate_prompt(ocr_output):
    return f"""
        ocr output: {ocr_output}

        Extract invoice details in this JSON format:
        {{
            'product_Name': ['value1', 'value2'],
            'order_number': 'value',
            'order_cancel_number': null,
            'order_date_and_time': 'value',
            'delivery_date_and_time': 'value',
            'product_price': ['value1', 'value2'],
            'total_amount': 'value'
        }}

        Rules:
        1. Convert Arabic numerals (٠-٩) to standard numbers
        2. Return null for missing fields
        3. Use exact JSON format - no additional text
        4. For total_amount, include all tax/discount calculations
        5. Do not include any introduction or conclusion in the output and no code
    """

@app.post("/extract_invoice/")
async def extract_invoice(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

       
        ocr_output = process_arabic_ocr(img_np)

        prompt = generate_prompt(ocr_output)

 
        completion = client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [{"type": "text", "text": prompt}]
            }],
            model=model
        )

        result = completion.choices[0].message.content
        return JSONResponse(content={"invoice_data": result})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
