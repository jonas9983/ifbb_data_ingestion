"""
Full Workflow:

Extract data from insta using Apify and save to instagram_downloads organized by athlete insta ID -> 
Run face_recognizer on all frames and find those that only have one face (presuming only the athlete will be there). Save results to validation directory ->
Validate faces and move them to database. Build the faces database

Run the main script from the athlete_tracking directory on images from the competition that we want to analyze.
"""