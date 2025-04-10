@echo off
cd %~dp0
python visualize_example.py ../../data/raw_pictures/turkana_2/37a1de7d-527b-4d56-823d-0e1f3b53a2ad/1517498265152.jpg --model_path ../../data/shape_predictor_68_face_landmarks.dat --output_dir visualization_output
pause 