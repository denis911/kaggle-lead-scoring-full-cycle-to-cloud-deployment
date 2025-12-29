import requests
import json

def run_local_test():
    url = "http://127.0.0.1:8000/predict"
    
    # Sample data matching LeadData schema
    payload = {
        "Lead Origin": "Landing Page Submission",
        "Lead Source": "Direct Traffic",
        "Do Not Email": "No",
        "TotalVisits": 2.0,
        "Total Time Spent on Website": 1532.0,
        "Page Views Per Visit": 2.0,
        "Last Activity": "Email Opened",
        "Country": "India",
        "Specialization": "Marketing Management",
        "What is your current occupation": "Unemployed",
        "City": "Mumbai",
        "A free copy of Mastering The Interview": "Yes",
        "Last Notable Activity": "Email Opened"
    }

    print(f"--- Sending Lead to {url} ---")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Lead Score: {result['lead_score']}")
        print(f"Hot Lead: {result['is_hot_lead']}")
        print(f"Conversion Prob: {result['conversion_probability']}")
        print("-------------------------------")
        
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to API: {e}")
        print("Make sure the server is running with: uv run uvicorn app:app --reload")

if __name__ == "__main__":
    run_local_test()
