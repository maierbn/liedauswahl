import requests
from concurrent.futures import ThreadPoolExecutor
import os
import random
import json

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import time
import datetime

import asyncio
from typing import List, Callable
import psutil

# define task queue
task_queue: asyncio.Queue = asyncio.Queue()


async def task_worker():
    while True:
        func = await task_queue.get()
        try:
            await func()
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error during task execution: {e}")
        finally:
            task_queue.task_done()

app = FastAPI()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(task_worker())


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
# Separate models for the two AI stages.
SCORING_MODEL = "gpt-6-luna"       # Fast, inexpensive evaluation of all songs
RANKING_MODEL = "gpt-6.1-terra"    # Stronger reasoning for final pairwise ranking
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

# Shared API handler: finite retries for transient failures, explicit errors for HTTP 400.
def call_openai(data, max_retries=5):
    url = f"{base_url.rstrip('/')}/chat/completions"
    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=data, timeout=120)
        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt == max_retries - 1:
                raise RuntimeError(f"OpenAI network error after retries: {exc}") from exc
            time.sleep(min(2 ** attempt, 30) + random.random())
            continue

        try:
            result_json = response.json()  # Parse BEFORE inspecting/printing it.
        except ValueError as exc:
            raise RuntimeError(
                f"OpenAI returned non-JSON response HTTP {response.status_code}: "
                f"{response.text[:2000]}"
            ) from exc

        if not response.ok:
            error = result_json.get("error", result_json) if isinstance(result_json, dict) else result_json
            print(f"OpenAI API ERROR: HTTP {response.status_code}: {error}", flush=True)
            if response.status_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                retry_after = response.headers.get("Retry-After")
                try:
                    delay = max(0, min(float(retry_after), 120)) if retry_after else min(2 ** attempt, 30) + random.random()
                except ValueError:
                    delay = min(2 ** attempt, 30) + random.random()
                time.sleep(delay)
                continue
            raise RuntimeError(f"OpenAI API error (HTTP {response.status_code}): {error}")

        choices = result_json.get("choices") if isinstance(result_json, dict) else None
        if not choices:
            raise RuntimeError(f"OpenAI response missing choices: {result_json}")
        choice = choices[0]
        content = choice.get("message", {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError(f"OpenAI returned empty content (finish_reason={choice.get('finish_reason')}); "
                               f"increase max_completion_tokens if reasoning uses the budget: {result_json}")
        return content.split("</think>")[-1].strip()
    raise RuntimeError("OpenAI request failed after retries")


print(f"\n\nStarting Backend")

def process_songs(selected_books, topic, receiver_email):
    print(f"AI models: scoring={SCORING_MODEL}, ranking={RANKING_MODEL}")
    # load song texts
    print(f"New request: topic={topic}, receiver_email={receiver_email}, selected_books={selected_books}\n")

    import numpy as np
    import pandas as pd
    import json
    import requests
    import os
    import json

    # Create a DataFrame from the extracted data
    df_songs = pd.read_parquet("songs_with_eg.parquet")

    # filter
    df_songs = df_songs[df_songs.book.isin(selected_books)]

    if len(df_songs) == 0:
        selected_books = ["Feiert Jesus 2", "Feiert Jesus 3", "Feiert Jesus 4", "Feiert Jesus 5", "Feiert Jesus 6", "Feiert Jesus Best Of"]
        df_songs = pd.read_parquet("songs_with_eg.parquet")
        df_songs = df_songs[df_songs.book.isin(selected_books)].copy()

    # unify location
    df_songs['location'] = ""
    for index, row in df_songs.iterrows():
        location = ""
        for i, other_row in df_songs[df_songs["title"] == row["title"]].reset_index(drop=True).iterrows():
            if i != 0:
                location += "; "
            
            location += f"{other_row['book'].replace('Feiert Jesus ', 'FJ').replace('Best Of', 'B')} {other_row['number']}"
        df_songs.loc[index, 'location'] = location

    df_songs = df_songs.drop_duplicates(subset="title")
    print(f"total number of songs: {len(df_songs)}")

    if len(df_songs) > 550:
        print(f"sample 550")
        df_songs = df_songs.sample(n=550)

    def get_score(row, topic):
        topic = topic.replace("\"", "")

        message = f"""Bewerte folgendes Lied, wie gut es zum Thema des Gottesdienstes passt. Liefere einen Score zwischen 0 (passt nicht) und 1 (passt sehr gut).
        Deine Antwort soll NUR aus dem string "score=0.5" bestehen, ohne Erklärung (wobei statt 0.5 der Score stehen soll).
        Thema des Gottesdienstes: \"{topic}\"
        Lied:
        """
        message += f"Titel: \"{row['title']}\", Liedtext: {row['lyrics']}\n\n"

        data = {
            "model": SCORING_MODEL,
            "messages": [{"role": "user", "content": message}],
            "reasoning_effort": "low",
            "max_completion_tokens": 300,
        }
        result_message = call_openai(data)

        score = 0
        if "score=" in result_message:
            p = result_message.find("score=") + len("score=")
            score = float("".join([s for s in result_message[p:p+10] if s in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "."]]))
        
        #if score > 0.8:
        #    print(f"{score}: \"{row['title']}\", {row['book']}, S.{row['number']}")

        return score

    # Function to process a single row
    def process_row(row, topic):
        score = get_score(row, topic)
        return row['title'], row['location'], row['lyrics'], score

    # Parallel execution using ProcessPoolExecutor
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda row: process_row(row, topic), [row for _, row in df_songs.iterrows()]))

    # Convert results to DataFrame
    df_scores = pd.DataFrame(results, columns=['title', 'location', 'lyrics', 'score'])

    df_scores = df_scores.sort_values("score", ascending=False)
    #print(df_scores)
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

        data = {
            "model": RANKING_MODEL,
            "messages": [{"role": "user", "content": message}],
            "reasoning_effort": "medium",
            "max_completion_tokens": 600,
        }
        result_message = call_openai(data)

        if "[1]" in result_message:
            return True
        if "[2]" in result_message:
            return False
        raise ValueError(f"Unexpected comparison output: {result_message!r}")

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
        """Sort both halves safely. This shortlist is small; processes are not necessary."""
        return merge_sort_df(left_half, topic), merge_sort_df(right_half, topic)

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
        password = os.environ.get('SMTP_PASSWORD')
        if not password:
            raise RuntimeError('Set SMTP_PASSWORD environment variable before sending mail')

        message = MIMEMultipart()
        message['From'] = sender_email
        message['To'] = receiver_email
        message['Subject'] = 'Liedauswahl'

        message.attach(MIMEText(html_content, 'html'))
        message['Bcc'] = bcc_email

        context = ssl.create_default_context()

        with smtplib.SMTP_SSL('mx.freenet.de', port) as server:
            server.login(sender_email, password)
            server.sendmail(sender_email, [receiver_email, bcc_email], message.as_string())

    send_mail(receiver_email, html_content=html_output)
    print(f"sent mail to {receiver_email}.")

    # Create filename with current datetime
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    topic_shortened = topic.replace("\n", "").replace("\\","").replace("/", "").replace("\r", "").replace("\t", "_").replace(" ", "_").replace(".", "_")
    topic_shortened = topic_shortened[0:min(50,len(topic_shortened))]
    filename = f"logs/{timestamp}_{topic_shortened}.html"

    # Write to the HTML file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_output.replace("<body>", f"<body><p>Receiver: {receiver_email}</p>\n"))

    # Return the sorted DataFrame as a dictionary
    return df_sorted.to_dict(orient="records")

@app.post("/process_songs")
async def process_songs_endpoint(request: SongRequest):
    selected_books = request.selected_books
    topic = request.topic
    receiver_email = request.receiver_email
    
    # Define a function to run in the background
    async def background_process():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, process_songs, selected_books, topic, receiver_email)
   
    await task_queue.put(background_process)

    n_pending_tasks = task_queue.qsize()

    # Return a response immediately
    result = {
        "status": "success",
        "message": f"Der Vorgang wurde gestartet, Platz {n_pending_tasks} in der Warteschlange. In ein paar Minuten wird das Ergebnis an `{receiver_email}` geschickt."
    }
    print(result["message"])
    return result

@app.get("/")
async def test_endpoint():
    return {"message": "API is reachable"}


@app.post("/test")
async def test_endpoint():
    return {"message": "API is reachable"}


@app.get("/server_status")
async def server_status():
    # Memory usage
    memory_info = psutil.virtual_memory()
    memory_used_gb = (memory_info.total - memory_info.available) / (1024 ** 3)

    # Queue length
    n_pending_tasks = task_queue.qsize()

    # Build message
    anfragen = "Anfrage" if n_pending_tasks == 1 else "Anfragen"
    message = f"(Warteschlange: {n_pending_tasks} {anfragen}, aktuelle Speicherauslastung: {memory_used_gb:.2f} von 1 GB)"
    return {"message": message}

