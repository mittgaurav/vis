#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download VIS Phase 1 dataset directly from shared Google Drive folder
Usage: python download_vis_phase1.py
Files saved to: ./VIS/phase_1/
"""

import os
import io
import zipfile
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from tqdm import tqdm
import pickle

# CONFIGURATION
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
SOURCE_FOLDER_ID = '151jsqDzHIsW892H4vpPoLS3W2IkZrh5F'
BASE_DEST_PATH = './VIS/phase_1'


def authenticate():
    """Authenticate and return Drive service"""
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)  # ← Download from Google Cloud Console
            creds = flow.run_local_server(port=0)

        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return build('drive', 'v3', credentials=creds)


def download_and_unzip(service, file_id, file_name, extract_to):
    """Download and unzip file"""
    request = service.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    with tqdm(desc=f"Downloading {file_name}", unit="B", unit_scale=True) as pbar:
        while done is False:
            status, done = downloader.next_chunk()
            pbar.update(status.progress().resumable_progress)

    fh.seek(0)
    with zipfile.ZipFile(fh, 'r') as z:
        z.extractall(extract_to)
    return True


def main():
    # Setup
    os.makedirs(BASE_DEST_PATH, exist_ok=True)
    service = authenticate()

    print("🚀 Downloading VIS Phase 1 dataset...")

    # 1. Annotations
    print("\n📋 Downloading annotations...")
    annot_dest = os.path.join(BASE_DEST_PATH, 'annotations')
    os.makedirs(annot_dest, exist_ok=True)

    query = f"'{SOURCE_FOLDER_ID}' in parents and name='annotations.zip'"
    results = service.files().list(q=query, fields="files(id,name)").execute()
    files = results.get('files', [])

    if files:
        download_and_unzip(service, files[0]['id'], files[0]['name'], annot_dest)
        print("✅ Annotations complete")

    # 2. Train data
    print("\n🎯 Downloading train data...")
    train_dest = os.path.join(BASE_DEST_PATH, 'train')
    os.makedirs(train_dest, exist_ok=True)

    # Find train folder
    query = f"'{SOURCE_FOLDER_ID}' in parents and name='train' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id,name)").execute()
    train_folder_id = results.get('files', [{}])[0].get('id')

    if train_folder_id:
        # Find all zips in train folder
        query = f"'{train_folder_id}' in parents and name contains '.zip'"
        results = service.files().list(q=query, fields="files(id,name)").execute()
        zips = results.get('files', [])

        for zip_file in tqdm(zips, desc="Train zips"):
            folder_name = os.path.splitext(zip_file['name'])[0]
            target_dir = os.path.join(train_dest, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            download_and_unzip(service, zip_file['id'], zip_file['name'], target_dir)

    # 3. Pub test data
    print("\n🧪 Downloading pub_test data...")
    pub_test_dest = os.path.join(BASE_DEST_PATH, 'pub_test')
    os.makedirs(pub_test_dest, exist_ok=True)

    # Find pub_test folder
    query = f"'{SOURCE_FOLDER_ID}' in parents and name='pub_test' and mimeType='application/vnd.google-apps.folder'"
    results = service.files().list(q=query, fields="files(id,name)").execute()
    pub_test_folder_id = results.get('files', [{}])[0].get('id')

    if pub_test_folder_id:
        # Find all zips in pub_test folder
        query = f"'{pub_test_folder_id}' in parents and name contains '.zip'"
        results = service.files().list(q=query, fields="files(id,name)").execute()
        zips = results.get('files', [])

        for zip_file in tqdm(zips, desc="Pub_test zips"):
            folder_name = os.path.splitext(zip_file['name'])[0]
            target_dir = os.path.join(pub_test_dest, folder_name)
            os.makedirs(target_dir, exist_ok=True)
            download_and_unzip(service, zip_file['id'], zip_file['name'], target_dir)

    print(f"\n🎉 COMPLETE! Dataset ready at: {BASE_DEST_PATH}")


if __name__ == "__main__":
    main()
