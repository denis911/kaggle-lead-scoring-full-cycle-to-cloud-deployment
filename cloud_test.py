import requests
import json

def run_cloud_test():
    # Render handles HTTPS on port 443 automatically, so no port needed.
    url = "https://kaggle-lead-scoring-full-cycle-to-cloud.onrender.com/predict"
    
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

    print(f"--- Sending Lead to Cloud API: {url} ---")
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print(f"Status Code: {response.status_code}")
        print(f"Lead Score: {result['lead_score']}")
        print(f"Hot Lead: {result['is_hot_lead']}")
        print(f"Conversion Prob: {result['conversion_probability']}")
        print("Final Verification: SUCCESS ✅")
        print("-------------------------------")
        
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response.status_code == 404:
            print("Check if the URL path /predict is correct.")
    except Exception as e:
        print(f"Error connecting to Cloud API: {e}")

if __name__ == "__main__":
    run_cloud_test()
