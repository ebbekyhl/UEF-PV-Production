from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os
import json
import pandas as pd
import time

def initial_cleaning(download_dir):
    # if .tmp files exist in the download directory, delete them
    tmp_files = [f for f in os.listdir(download_dir) if f.endswith('.tmp')]
    for tmp_file in tmp_files:
        os.remove(os.path.join(download_dir, tmp_file))
        print(f"Deleted temporary file: {tmp_file}")

def read_tmp(year_month, download_dir):
    # find files ending with .tmp in the download directory
    tmp_file = [f for f in os.listdir(download_dir) if f.endswith('.tmp')][0]

    filepath = os.path.join(download_dir, tmp_file)

    # Wait until the file is fully downloaded and renamed (you can customize the logic)
    time.sleep(5)  # wait for download completion

    # Read the .tmp file content
    with open(filepath, "r") as f:
        data_str = f.read()

    # If it's JSON, parse it
    try:
        data_dict = json.loads(data_str)
    except json.JSONDecodeError:
        # If it's a Python dict-like string, you can safely use ast.literal_eval
        import ast
        data_dict = ast.literal_eval(data_str)

    # Convert to pandas DataFrame
    df = pd.DataFrame(data_dict["data"])

    # Save to .csv
    csv_file_path = f"{download_dir}/PV_production_Aarhus_{year_month}.csv"
    print(csv_file_path)
    df[["date","Ep"]].to_csv(csv_file_path, index=False, header=True, sep=";")
    
    # delete .tmp file
    os.remove(download_dir + "/" +  tmp_file)

def download_pv_data(year_month,download_dir):

    # delete .tmp files if they exist
    initial_cleaning(download_dir)

    # URL for the PV data download
    url = "https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month=" + year_month

    # Preferences for Chrome download settings
    prefs = {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    }

    # Set up headless Chrome options
    options = Options()
    # options.headless = True # Recommended by ChatGPT
    options.add_experimental_option("prefs", prefs)
    options.add_argument("--headless=new")  # Required for headless downloads in Chrome 96+
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    # options.add_argument("--disable-dev-shm-usage") # Recommended by ChatGPT

    # Set up driver
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    try:
        driver.get(#"https://dashboard.aurora-h2020.eu/en-GB/pv-data?site=DK01&month={year_month}")
                url)

        # Wait for page to load and button to appear
        wait = WebDriverWait(driver, 20)
        download_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[contains(., 'Download')]")))

        download_button.click()
        print("Download initiated.")

    except Exception as e:
        print("Error:", e)

    finally:
        driver.quit()

    read_tmp(year_month, download_dir)
    
    print("Download and processing completed. Data saved to CSV.")