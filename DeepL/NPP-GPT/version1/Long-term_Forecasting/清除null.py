file_path = 'E:/Document/CodeSpace/Study/DeepL/NPP-GPT/version1/Long-term_Forecasting/data_provider/data_loader.py'

# Check for null bytes in the file
with open(file_path, 'rb') as file:
    content = file.read()
    if b'\x00' in content:
        print("Null bytes found in file. Cleaning up...")

        # Remove null bytes
        clean_content = content.replace(b'\x00', b'')
        
        # Write the cleaned content back to the file
        with open(file_path, 'wb') as clean_file:
            clean_file.write(clean_content)
        print("File cleaned successfully.")
    else:
        print("No null bytes found in file.")
