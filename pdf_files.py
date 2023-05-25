def get_fullpath_of_pdf_files(path):
  import os

  file_names = os.listdir(path)
  
  return [path + '/'+file_name for file_name in file_names if file_name.endswith('.pdf')]