# Implementation - Data Crawlers Dashboard

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
  - [Key Components](#key-components)
    - [app.py](#apppy)
    - [dynamic_updates.js](#dynamic_updatesjs)
    - [styles.css](#stylescss)
    - [home.html](#homehtml)
    - [dashboard.html](#dashboardhtml)
- [API Endpoints](#api-endpoints)

## Introduction
The **Data Crawlers Dashboard** is a comprehensive web-based application designed to monitor, manage, and visualize data crawling activities from platforms like Reddit and 4chan. It provides users with an intuitive interface to oversee multiple data crawlers, track their performance, analyze collected data, and gain insights through various interactive charts and metrics.

## Features
- **Question Interface:** Integrates a large language model from Hugging face (EleutherAI/gpt-neo-1.3B) for generating AI responses to user queries about the data from 4chan and reddit.
- **Real-time Monitoring:** View the status and performance metrics of active data crawlers.
- **Dynamic Updates:** Automatic refreshing of dashboard data without needing to reload the page.
- **Interactive Charts:** Visualize data through bar charts, heatmaps, histograms, and more.
- **User-Friendly Interface:** Clean and responsive UI built with HTML, CSS, and JavaScript.
- **Platform Selection:** Choose between Reddit and 4chan analyses seamlessly.
- **Toxicity Analysis:** Analyze the toxicity of threads on 4chan.
- **Secure Environment:** Utilizes virtual environments to manage dependencies safely.

## Architecture
The application follows a modular architecture comprising the following components:

- **Backend:** Built with Python and Flask, handling data processing, API endpoints, and serving the frontend.
- **Frontend:** Developed using HTML, CSS, and JavaScript for the user interface.
- **Templates:** Uses Jinja2 templating engine for rendering dynamic content.
- **Static Files:** Contains CSS and JavaScript files for styling and interactivity.
- **Database:** Utilizes MongoDB for storing and retrieving data from Reddit and 4chan.

## Installation

### Prerequisites
- **Python 3.6+**
- **Git**
- **MongoDB Instance**
- **Virtual Environment Tool (e.g., `venv`)**

### Steps

1. **Clone the Repository but not needed as it is already in the VM**
    ```bash
    git clone https://github.com/yourusername/data-crawlers-dashboard.git
    cd data-crawlers-dashboard
    ```

2. **Navigate to the Data Crawlers Directory**
    ```bash
    cd /home/Data_Crawlers/
    ```

3. **Activate the Virtual Environment**
    ```bash
    source web_ui/bin/activate
    ```
    - **To deactivate:**
      ```bash
      deactivate
      ```

4. **Navigate to the Project Directory**
    ```bash
    cd /home/Data_Crawlers/project-3-implementation-elbaf
    ```

5. **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

6. **Configure Environment Variables**
    - Create a `.env` file in the project root directory or use the existing .env.
    - Add the following line with your MongoDB URL:
      ```env
      MONGO_DB_URL=your_mongodb_connection_string
      ```
    - **Note:** Ensure that the `.env` file is **not** pushed to GitHub for security reasons. The `.gitignore` file is configured to exclude `.env`. 

## Usage

1. **Start the Application**
    ```bash
    python3 app.py
    ```

2. **Access the Dashboard**
    - Open your web browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Select Platform**
    - On the home page, choose between **Reddit Analysis** or **4chan Analysis** to view respective dashboards.

## Project Structure
```bash
project-3-implementation-elbaf/
├── .gitignore
├── CREDITS.md
├── FEEDBACK.md
├── HONESTY.md
├── LLM_images.png
├── Question_to_ask.txt
├── README.md
├── app.py
├── requirements.txt
├── static/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── dynamic_updates.js
├── templates/
│   ├── dashboard.html
│   └── home.html
```

## Key Components

### app.py
The main application file that initializes and runs the Flask server. It defines various API endpoints for both Reddit and 4chan data analysis, handles database interactions with MongoDB, and integrates a language model for generating AI responses to user queries.

**Highlights:**
- **Flask App Initialization:** Sets up the Flask application with CORS support.
- **MongoDB Configuration:** Connects to MongoDB using the provided `MONGO_DB_URL`.
- **API Endpoints:** Multiple endpoints to fetch and process data for different charts and analyses.
- **AI Integration:** Uses the GPT-Neo model to generate responses based on user prompts.

### dynamic_updates.js
Handles dynamic updates and interactivity for the Reddit dashboard. It manages chart rendering, data fetching, and user interactions such as filtering and asking questions.

**Highlights:**
- **Chart Initialization:** Sets up various Chart.js charts for visualizing data.
- **Data Fetching:** Utilizes AJAX and jQuery to fetch data from API endpoints and update charts accordingly.
- **User Interactions:** Manages form submissions for asking questions and updating charts based on user input.

### styles.css
Contains all the CSS styles for the frontend, ensuring a responsive and visually appealing user interface.

**Highlights:**
- **Responsive Design:** Ensures the dashboard looks good on various screen sizes.
- **Theming:** Implements a dark theme with vibrant accent colors for charts and interactive elements.
- **Component Styling:** Styles for the sidebar, main content, cards, buttons, and footer.

### home.html
The landing page where users can select the platform (Reddit or 4chan) they wish to analyze.

**Highlights:**
- **Platform Selection:** Provides buttons for users to choose between Reddit and 4chan analyses.
- **Animations:** Uses Animate.css for smooth entry animations.
- **Responsive Layout:** Centers content both vertically and horizontally for an optimal user experience.

### dashboard.html
The main dashboard interface that displays various charts and metrics based on the selected platform (Reddit or 4chan).

**Highlights:**
- **Dynamic Content Rendering:** Uses Jinja2 to render content based on the selected platform.
- **Interactive Charts:** Integrates multiple Chart.js charts for data visualization.
- **Form Handling:** Includes forms for date range selection, subreddit selection, and question submission.
- **Toxicity Analysis:** New feature to analyze and visualize thread toxicity on 4chan.
- **Footer:** Consistent footer across all pages.

## API Endpoints

### 4chan Endpoints
- `/get_4chan_thread_activity` (GET): Retrieves thread creation and reply activity over time.
- `/get_4chan_reply_frequency` (GET): Calculates average replies per thread for each board.
- `/get_4chan_thread_lifespan` (GET): Determines the lifespan of the most popular threads.
- `/get_4chan_popular_vs_unpopular` (GET): Compares popular and unpopular threads based on a reply threshold.
- `/get_4chan_hourly_heatmap` (GET): Generates a heatmap of hourly activity.
- `/get_4chan_popularity_histogram` (GET): Creates a histogram of thread popularity.
- `/get_4chan_replies_vs_posts` (GET): Compares replies to original posts by board.
- `/get_4chan_daily_data` (GET): Fetches daily post data for a specific board.
- `/get_4chan_hourly_data` (GET): Fetches hourly post data for a specific board.
- `/get_4chan_toxicity_data` (GET): Analyzes thread toxicity by class.

### Reddit Endpoints
- `/get_comments_data` (GET): Retrieves comments data, including total comments, peak hour, deleted posts, active users, and average comments per hour.
- `/get_reddit_submissions` (GET): Fetches the number of submissions per day within a date range.
- `/get_deleted_posts` (GET): Retrieves the count of deleted posts by day.
- `/get_comment_upvotes` (GET): Gets the distribution of comment upvotes.
- `/ask_question` (POST): Allows users to ask questions about the data and receive AI-generated responses.
