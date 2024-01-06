import webbrowser
import os
import scrap


def open_streamlit_on_chrome():
    streamlit_command = "streamlit run st_fast.py"
    os.system(streamlit_command)

    chrome_path = "C:/Program Files/Google/Chrome/Application/chrome.exe"
    url_to_open = "http://localhost:8501"

    webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
    webbrowser.get('chrome').open(url_to_open)


if __name__ == "__main__":
    scrap.update_df()
    scrap.update_bench()
    open_streamlit_on_chrome()
    
