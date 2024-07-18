import os
import pickle
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import Flow, InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

class DriveUploader:
    def __init__(self, credentials_path: str, token_path: str = None, use_service_account: bool = False):
        self.credentials_path = credentials_path
        self.token_path = token_path
        self.use_service_account = use_service_account
        self.service = self._authenticate()

    def _authenticate(self):
        if self.use_service_account:
            return self._authenticate_service_account()
        else:
            return self._authenticate_oauth()

    def _authenticate_service_account(self):
        credentials = service_account.Credentials.from_service_account_file(
            self.credentials_path,
            scopes=['https://www.googleapis.com/auth/drive.file']
        )
        return build('drive', 'v3', credentials=credentials)

    def _authenticate_oauth(self):
        creds = None
        if self.token_path and os.path.exists(self.token_path):
            with open(self.token_path, 'rb') as token:
                creds = pickle.load(token)
        
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, ['https://www.googleapis.com/auth/drive.file'])
                creds = flow.run_local_server(port=0)
            
            if self.token_path:
                with open(self.token_path, 'wb') as token:
                    pickle.dump(creds, token)

        return build('drive', 'v3', credentials=creds)

    def create_folder_tree(self, base_folder_id: str, path: str) -> str:
        current_folder_id = base_folder_id
        path_parts = path.split(os.sep)
        
        for part in path_parts:
            query = f"name = '{part}' and '{current_folder_id}' in parents and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
            results = self.service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
            folders = results.get('files', [])
            
            if folders:
                current_folder_id = folders[0]['id']
            else:
                folder_metadata = {
                    'name': part,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [current_folder_id]
                }
                folder = self.service.files().create(body=folder_metadata, fields='id').execute()
                current_folder_id = folder.get('id')
        
        return current_folder_id

    def upload_file(self, file_path: str, parent_folder_id: str):
        file_name = os.path.basename(file_path)
        file_metadata = {
            'name': file_name,
            'parents': [parent_folder_id]
        }
        media = MediaFileUpload(file_path, resumable=True)
        file = self.service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f'Uploaded file: {file_name} (ID: {file.get("id")})')

    def upload_folder(self, local_folder: str, drive_base_folder_id: str):
        for root, dirs, files in os.walk(local_folder):
            relative_path = os.path.relpath(root, local_folder)
            if relative_path == '.':
                current_folder_id = drive_base_folder_id
            else:
                current_folder_id = self.create_folder_tree(drive_base_folder_id, relative_path)
            
            for file in files:
                file_path = os.path.join(root, file)
                self.upload_file(file_path, current_folder_id)