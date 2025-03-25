import requests
import os
#pull submissions from kobo for this form
turkana_meron_survey = 537
marsabit_meron_survey = 513
isiolo_meron_survey = 604
tana_river_meron_survey = 551

form_id = tana_river_meron_survey
form_name = 'tana_river_meron_survey'
folder_path = 'C:/Users/Hussein Lightwalla/Desktop/meron_pics'

form_folder = '{}/{}'.format(folder_path, form_name)
if not os.path.exists(form_folder):
    os.makedirs(form_folder)

r = requests.get('https://kobo.kimetrica.com/kobocat/api/v1/data/{}'.format(form_id), auth=('meron_surveys', 'M3ron2018'))
r_json = r.json()
#print (r_json)
counter = 0
for submission in r_json:
    #if counter > 5:
    #    break
    if len(submission['_attachments']) > 0:
        image_dir = '{}/{}'.format(form_folder, submission['_uuid'])
        if not os.path.exists(image_dir):
            os.makedirs(image_dir)
        for image_item in submission['_attachments']:
            counter = counter + 1
            complete_download_url = 'https://kobo.kimetrica.com{}'.format(image_item['download_url'])
            print(complete_download_url)
            file_name = complete_download_url.rsplit('/')[-1]
            file_name = '{}/{}'.format(image_dir, file_name)
            if not os.path.exists(file_name):
                f = open(file_name, 'wb')
                f.write(requests.get(complete_download_url).content)
                f.close()
