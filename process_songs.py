import requests
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time

app = FastAPI()

# Add CORS middleware
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],  # Allows all HTTP methods
   allow_headers=["*"],  # Allows all headers
)

# Define request body structure using Pydantic
class SongRequest(BaseModel):
    selected_books: list
    topic: str
    receiver_email: str


# parameters
selected_books = ["Feiert Jesus 4", "Feiert Jesus 5", "Feiert Jesus Best Of"]
topic = "Jesus Christus nachfolgen heißt aushalten, dass bei Gott andere Maßstäbe von Gerechtigkeit, von Ruhm und Ehre gelten als in der Welt. Nicht die eigene Leistung zählt, sondern Gottes Gnade."
receiver_email = "maierbn+tester@gmail.com"

base_url = "http://localhost:8000/v1"
model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"


# GPT
base_url = "https://api.openai.com/v1/"
model = "gpt-4o-mini"
with open("token", 'r') as file:
    token = file.read().strip()

# DeepSeek
#base_url = "https://api.deepseek.com"
#model = "deepseek-reasoner"
#token = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

print(headers)

def process_songs(selected_books, topic, receiver_email):
    # load song texts

    import numpy as np
    import pandas as pd
    import json
    import requests
    import os
    import json

    # Create a DataFrame from the extracted data
    df_songs = pd.read_parquet("songs.parquet")

    # filter
    df_songs = df_songs[df_songs.book.isin(selected_books)]

    if len(df_songs) == 0:
        selected_books = ["Feiert Jesus 2", "Feiert Jesus 3", "Feiert Jesus 4", "Feiert Jesus 5", "Feiert Jesus 6", "Feiert Jesus Best Of"]

    # unify location
    df_songs['location'] = ""
    for index, row in df_songs.iterrows():
        location = ""
        for i, other_row in df_songs[df_songs["title"] == row["title"]].reset_index(drop=True).iterrows():
            if i != 0:
                location += "; "
            
            location += f"{other_row['book'].replace('Feiert Jesus ', 'FJ').replace('Best Of', 'B')}, Nr.{other_row['number']}"
        df_songs.loc[index, 'location'] = location

    df_songs = df_songs.drop_duplicates(subset="title")

    def get_score(row, topic):
        topic = topic.replace("\"", "")

        message = f"""Bewerte folgendes Lied, wie gut es zum Thema des Gottesdienstes passt. Liefere einen Score zwischen 0 (passt nicht) und 1 (passt sehr gut).
        Deine Antwort soll NUR aus dem string "score=0.5" bestehen, ohne Erklärung (wobei statt 0.5 der Score stehen soll).
        Thema des Gottesdienstes: \"{topic}\"
        Lied:
        """
        message += f"Titel: \"{row['title']}\", Liedtext: {row['lyrics']}\n\n"

        url = f"{base_url}/chat/completions"
        data = {
        "messages": [
            {
            "content": message,
            "role": "user",
            "name": "string"
            },
        ],
        "temperature": 0.7,
        "model": model
        }
        response = requests.post(url, headers=headers, json=data)
        result_json = response.json()

        if 'error' in result_json:
            if "Rate limit reached" in result_json['error']['message']:
                time.sleep(10)
                return get_score(row, topic)
            raise RuntimeError(result_json['error']['message'])
        result_message = result_json['choices'][0]['message']['content']

        if "</think>" in result_message:
            p1 = result_message.find("</think>") + len("</think>")
            result_message = result_message[p1:]

        score = 0
        if "score=" in result_message:
            p = result_message.find("score=") + len("score=")
            score = float("".join([s for s in result_message[p:p+10] if s in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]]))
        
        if score > 0.8:
            print(f"{score}: \"{row['title']}\", {row['book']}, S.{row['number']}")

        return score

    # Function to process a single row
    def process_row(row, topic):
        score = get_score(row, topic)
        return row['title'], row['location'], row['lyrics'], score

    # Parallel execution using ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=50) as executor:
        results = list(executor.map(lambda row: process_row(row, topic), [row for _, row in df_songs.iterrows()]))

    # Convert results to DataFrame
    df_scores = pd.DataFrame(results, columns=['title', 'location', 'lyrics', 'score'])

    df_scores = df_scores.sort_values("score", ascending=False)
    print(df_scores)
    threshold = 0.9
    df_selected = df_scores[df_scores.score>=threshold]

    for i in range(8):
        if len(df_selected) >= 15:
            break
        threshold -= 0.1
        df_selected = df_scores[df_scores.score>=threshold]
    print(f"selected {len(df_selected)} with threshold {threshold}.")

    # compare two songs
    def is_1_better(row1, row2, topic):
        topic = topic.replace("\"", "")

        message = f"""Welches der beiden folgenden Lieder passt besser zum Thema des Gottesdienstes? Antworte mit [1] oder [2].
        Deine Antwort soll NUR aus dem string "[1]" oder "[2]" bestehen, ohne Erklärung.
        Thema des Gottesdienstes: \"{topic}\"
        Lied [1]: Titel: "{row1['title']}", Liedtext: {row1['lyrics']}\n\n
        Lied [2]: Titel: "{row2['title']}", Liedtext: {row2['lyrics']}
        """

        url = f"{base_url}/chat/completions"
        data = {
        "messages": [
            {
            "content": message,
            "role": "user",
            "name": "string"
            },
        ],
        "temperature": 0.7,
        "model": model
        }
        response = requests.post(url, headers=headers, json=data)
        result_json = response.json()
        if 'error' in result_json:
            if "Rate limit reached" in result_json['error']['message']:
                time.sleep(10)
                return is_1_better(row1, row2, topic)
        result_message = result_json['choices'][0]['message']['content']
        
        #with open(f"outputs/compare_{row1['title']}_{row2['title']}.txt", "w") as file:
        #    file.write(message)
        #    file.write("\n\n----------------\n\n")
        #    file.write(result_message)

        if "</think>" in result_message:
            p1 = result_message.find("</think>") + len("</think>")
            result_message = result_message[p1:]

        if "[1]" in result_message:
            return True
        return False

    import pandas as pd
    import multiprocessing as mp

    def merge_sort_df(df, topic):
        """Sorts a DataFrame using parallel merge sort based on is_1_better function."""
        
        if len(df) <= 1:
            return df

        mid = len(df) // 2
        left_half = df.iloc[:mid]
        right_half = df.iloc[mid:]

        # Run sorting in parallel using separate processes
        left_sorted, right_sorted = parallel_sort(left_half, right_half, topic)

        # Merge the sorted halves
        return merge(left_sorted, right_sorted, topic)

    def parallel_sort(left_half, right_half, topic):
        """Sorts both halves in parallel using separate processes."""
        
        # Create shared queues
        queue_left = mp.Queue()
        queue_right = mp.Queue()

        # Define worker function
        def sort_and_store(df, queue):
            sorted_df = merge_sort_df(df, topic)
            queue.put(sorted_df.to_dict())  # Convert DF to dict before passing

        # Spawn separate sorting processes
        p1 = mp.Process(target=sort_and_store, args=(left_half, queue_left))
        p2 = mp.Process(target=sort_and_store, args=(right_half, queue_right))

        # Start processes
        p1.start()

        if len(right_half) < 15:
            p2.start()

        # Retrieve sorted results
        left_sorted = pd.DataFrame(queue_left.get())  # Convert dict back to DF

        if len(right_half) < 15:
            right_sorted = pd.DataFrame(queue_right.get())  # Convert dict back to DF
        else:
            right_sorted = right_half.copy()

        # Ensure processes finish
        p1.join()

        if len(right_half) < 15:
            p2.join()

        return left_sorted, right_sorted

    def merge(left, right, topic):
        """Merges two sorted DataFrames based on is_1_better function."""
        sorted_rows = []
        left_index, right_index = 0, 0

        while left_index < len(left) and right_index < len(right):
            row1 = left.iloc[left_index]
            row2 = right.iloc[right_index]

            if is_1_better(row1, row2, topic):  # If row1 is better, it comes first
                sorted_rows.append(row1.to_dict())  # Convert row to dict
                left_index += 1
            else:
                sorted_rows.append(row2.to_dict())  # Convert row to dict
                right_index += 1

        # Add remaining rows
        sorted_rows.extend(left.iloc[left_index:].to_dict('records'))
        sorted_rows.extend(right.iloc[right_index:].to_dict('records'))

        # Return a DataFrame (correctly structured)
        return pd.DataFrame(sorted_rows)

    df_sorted = merge_sort_df(df_selected, topic)


    # format result
    # Generate HTML table with collapsible lyrics in a new row
    html_output = """
    <html>
    <head>
        <style>
            table {
                border-collapse: collapse;
                width: 100%;
            }
            table, th, td {
                border: 1px solid black;
            }
            .lyrics {
                display: none;
                padding: 10px;
                background-color: #f4f4f4;
                border: 1px solid #ddd;
            }
            .collapsible {
                cursor: pointer;
                padding: 5px;
                text-align: left;
                background-color: #f1f1f1;
                border: 1px solid #ddd;
                width: 100%;
            }
            td {
                padding: 8px;
            }
        </style>
    </head>
    <body>

    <div>""" \
    + f"""
    <p>Thema: \"{topic}\"</p>
    <p>Ausgewählte Bücher: {', '.join(sorted(selected_books))}</p>
    </div><p>""" 


    # Add songs
    for i, (idx, row) in enumerate(df_sorted.iloc[:15].iterrows()):
        html_output += f"  {i+1}. {row['title']} ({row['location']})<br>"
    html_output += """
    </p>
    <table>
        <thead>
            <tr>
                <th></th>
                <th>Lied</th>
                <th>Buch</th>
            </tr>
        </thead>
        <tbody>
    """

    # Add table rows for each song
    for i, (idx, row) in enumerate(df_sorted.iloc[:15].iterrows()):
        html_output += f"""
        <tr> 
            <td>{i+1}</td>
            <td>{row['title']}</td>
            <td>{row['location']}</td>
        </tr>
        <tr class="lyrics-row">
            <td></td>
            <td colspan="2">{row['lyrics']}</td>
        </tr>
        """

    # Close the table and HTML tags
    html_output += """
        </tbody>
    </table>
    </body>
    </html>
    """

    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import ssl


    def send_mail(receiver_email, html_content):

        port = 465
        sender_email = 'be.c.m@freenet.de'
        bcc_email = 'maierbn@gmail.com'
        password = '7Z5W8=2WnUHuh.pe'

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = 'Liedauswahl'

        message.attach(MIMEText(html_content, 'html'))
        message['Bcc'] = bcc_email

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('mx.freenet.de', port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, receiver_email, message.as_string())

    send_mail(receiver_email, html_content=html_output)
    print(f"sent mail to {receiver_email}.")

    # Return the sorted DataFrame as a dictionary
    return df_sorted.to_dict(orient="records")

@app.post("/process_songs")
async def process_songs_endpoint(request: SongRequest, background_tasks: BackgroundTasks):
    selected_books = request.selected_books
    topic = request.topic
    receiver_email = request.receiver_email
    
    # Define a function to run in the background
    def background_process():
        try:
            process_songs(selected_books, topic, receiver_email)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during song processing: {e}")
    
    # Add the background task to be executed after the response is sent
    background_tasks.add_task(background_process)
    
    # Return a response immediately
    return {"status": "success", "message": f"Der Vorgang wurde gestartet. In ein paar Minuten wird das Ergebnis an {receiver_email} geschickt."}

@app.get("/")
async def test_endpoint():
    return {"message": "API is reachable"}


@app.post("/test")
async def test_endpoint():
    return {"message": "API is reachable"}
