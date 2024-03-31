# Accelerator-Meter-Analysis


Please find all the research details in research folder.

Steps for execution

    1. clone this repository with - git clone https://github.com/thirugnanam-tgs/accelerometer.git

    2. pip install -r requirements.txt

    3. Execute all cells in analysis.ipynb in research folder to create 4 models.
        1. model_all_data
        2. model_handled_outliers_removed_026
        3. model_without_026
        4. model_without_none

    4. streamlit run main.py - upload a file with x, y, and z acceleration as per the input data files provided (without headers)