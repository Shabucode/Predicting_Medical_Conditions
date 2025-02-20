import json
import requests

BASE_URL = "http://127.0.0.1:5000/"

# You can use this function to test your api
# Make sure the server is running locally on BASE_URL`
# or change the hardcoded localhost above
def test_predict():
    """
    Test the predict route with test data
    """
    test_description = {"description": "Dementia refers to a group of symptoms affecting memory, thinking, and social abilities, significantly interfering with a person's daily life. Alzheimer's disease is the most common cause of dementia."}
            #"Obsessive Compulsive Disorder (OCD) is a mental health condition characterized by persistent, intrusive thoughts (obsessions) and repetitive behaviors or mental acts (compulsions) performed to alleviate anxiety. Individuals with OCD often feel driven to perform these rituals, even when they realize they are excessive. OCD can significantly impact daily functioning and cause distress, but it is treatable with therapy and medication."
            #"Amyotrophic lateral sclerosis (ALS) is a progressive neurodegenerative disease that affects nerve cells in the brain and spinal cord, leading to muscle weakness, paralysis, and eventually respiratory failure. There is no cure for ALS, but treatment can help manage symptoms and improve quality of life.",
            #"Parkinson's disease is a progressive neurological disorder that primarily affects movement, leading to tremors, rigidity, and bradykinesia (slowness of movement). It results from the degeneration of dopamine-producing neurons in the brain.",
            #"Dementia refers to a group of symptoms affecting memory, thinking, and social abilities, significantly interfering with a person's daily life. Alzheimer's disease is the most common cause of dementia."
        
    print("Calling API with test description:")
    response = requests.post(f"{BASE_URL}/predict", data=json.dumps(test_description))
    print(test_description["description"])
    print("Response: ")
    print(response.status_code)
    print(response.json())
    assert response.status_code == 200


if __name__ == "__main__":
    test_predict()