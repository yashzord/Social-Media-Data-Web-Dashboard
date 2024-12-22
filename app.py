from flask import Flask, render_template, jsonify, request
import pymongo
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
from flask_cors import CORS
from bson import ObjectId
import time
import os
from datetime import datetime, timedelta
from collections import defaultdict
from dotenv import load_dotenv

app = Flask(__name__)
CORS(app)

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    raise ValueError("MONGO_DB_URL environment variable not set.")

client = pymongo.MongoClient(MONGO_DB_URL)
mongo_client = pymongo.MongoClient(MONGO_DB_URL)
db = mongo_client['reddit_data']
chan_db = mongo_client['4chan_data']
g_tv_collection = chan_db['g_tv_threads']
pol_collection = chan_db['pol_threads']

# New databases for toxicity analysis for 4chan
mod_db = mongo_client['4chan_moderate_data']
mod_coll = mod_db['g_tv_moderate_threads']

old_db = mongo_client['4chan_toxicity_old_threads']
old_coll = old_db['g_tv_old_threads']

collections = {
    "technology": db['posts'],
    "movies": db['posts'],
    "politics": db['reddit_politics']
}

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.route('/')
def home():
    """
    Rendering the home page where users can select a platform.
    """
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """
    Rendering the dashboard template.
    """
    media = request.args.get('media', 'reddit')
    return render_template('dashboard.html', platform=media)

def parse_dates(default_start='2024-11-01', default_end='2024-11-14', toxicity=False):
    """
    Parse start_date and end_date query parameters. 
    If `toxicity=True`, use 2024-11-15 as the default start date for 4chan.
    """
    if toxicity:
        default_start = '2024-11-15'
    
    start_date_str = request.args.get('start_date', default_start)
    end_date_str = request.args.get('end_date', default_end)
    start = datetime.strptime(start_date_str, '%Y-%m-%d')
    end = datetime.strptime(end_date_str, '%Y-%m-%d')
    return start, end


# ---------------------------4chan Endpoints---------------------------------

@app.route('/get_4chan_thread_activity', methods=['GET'])
def get_4chan_thread_activity():
    board = request.args.get('board', 'g')
    start, end = parse_dates()

    hourly_activity = defaultdict(lambda: {"threads_created": 0, "replies_received": 0})
    daily_activity = defaultdict(lambda: {"threads_created": 0, "replies_received": 0, "total_posts": 0})
    
    cursor = g_tv_collection.find({
        "board": board,
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        op_created_at = datetime.strptime(thread['original_post']['OP_Created_at'], '%Y-%m-%d %H:%M:%S')
        hour_key = op_created_at.replace(minute=0, second=0)
        day_key = op_created_at.date()

        hourly_activity[hour_key]["threads_created"] += 1
        daily_activity[day_key]["threads_created"] += 1
        daily_activity[day_key]["total_posts"] += 1

        for reply in thread.get("replies", []):
            reply_created_at = datetime.strptime(reply['Reply_Created_at'], '%Y-%m-%d %H:%M:%S')
            reply_hour_key = reply_created_at.replace(minute=0, second=0)
            reply_day_key = reply_created_at.date()

            hourly_activity[reply_hour_key]["replies_received"] += 1
            daily_activity[reply_day_key]["replies_received"] += 1
            daily_activity[reply_day_key]["total_posts"] += 1

    hourly_keys = sorted(hourly_activity.keys())
    labels = [dt.strftime('%Y-%m-%d %H:%M') for dt in hourly_keys]
    threads_created = [hourly_activity[k]["threads_created"] for k in hourly_keys]
    replies_received = [hourly_activity[k]["replies_received"] for k in hourly_keys]

    # Prepare daily data
    daily_keys = sorted(daily_activity.keys())
    daily_labels = [day.strftime('%Y-%m-%d') for day in daily_keys]
    daily_threads_created = [daily_activity[day]["threads_created"] for day in daily_keys]
    daily_replies_received = [daily_activity[day]["replies_received"] for day in daily_keys]
    daily_total_posts = [daily_activity[day]["total_posts"] for day in daily_keys]

    # Log daily data to the terminal
    for day_key in daily_keys:
        logging.info(f"Date: {day_key}, Threads Created: {daily_activity[day_key]['threads_created']}, "
                     f"Replies Received: {daily_activity[day_key]['replies_received']}, "
                     f"Total Posts: {daily_activity[day_key]['total_posts']}")

    return jsonify({
        "labels": labels,
        "threads_created": threads_created,
        "replies_received": replies_received,
        "daily_labels": daily_labels,
        "daily_threads_created": daily_threads_created,
        "daily_replies_received": daily_replies_received,
        "daily_total_posts": daily_total_posts
    })


@app.route('/get_4chan_reply_frequency', methods=['GET'])
def get_4chan_reply_frequency():
    start, end = parse_dates()
    reply_counts = defaultdict(list)

    cursor = g_tv_collection.find({
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        board = thread.get("board", "unknown")
        reply_counts[board].append(len(thread.get("replies", [])))

    boards = sorted(reply_counts.keys())
    avg_replies = [sum(reply_counts[b])/len(reply_counts[b]) if reply_counts[b] else 0 for b in boards]

    return jsonify({
        "boards": boards,
        "avg_replies": avg_replies
    })

@app.route('/get_4chan_thread_lifespan', methods=['GET'])
def get_4chan_thread_lifespan():
    start, end = parse_dates()
    most_popular_threads = {}

    cursor = g_tv_collection.find({
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        board = thread.get("board", "unknown")
        replies = thread.get("replies", [])
        if len(replies) > 0:
            replies_sorted = sorted(replies, key=lambda x: x['Reply_Created_at'])
            lifespan = (datetime.strptime(replies_sorted[-1]['Reply_Created_at'], '%Y-%m-%d %H:%M:%S') -
                        datetime.strptime(replies_sorted[0]['Reply_Created_at'], '%Y-%m-%d %H:%M:%S')).total_seconds()/3600
            current = most_popular_threads.get(board, (0, None, 0))
            if len(replies) > current[0]:
                most_popular_threads[board] = (len(replies), thread['thread_number'], lifespan)

    boards = sorted(most_popular_threads.keys())
    lifespans = [most_popular_threads[b][2] for b in boards]
    thread_ids = [most_popular_threads[b][1] for b in boards]

    return jsonify({
        "boards": boards,
        "lifespans": lifespans,
        "thread_ids": thread_ids
    })

@app.route('/get_4chan_popular_vs_unpopular', methods=['GET'])
def get_4chan_popular_vs_unpopular():
    start, end = parse_dates()
    threshold = int(request.args.get('threshold', 100))
    board_data = defaultdict(lambda: {"popular": 0, "unpopular": 0})

    cursor = g_tv_collection.find({
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        board = thread.get("board", "unknown")
        reply_count = len(thread.get("replies", []))
        if reply_count >= threshold:
            board_data[board]["popular"] += 1
        else:
            board_data[board]["unpopular"] += 1

    boards = sorted(board_data.keys())
    popular_counts = [board_data[b]["popular"] for b in boards]
    unpopular_counts = [board_data[b]["unpopular"] for b in boards]

    return jsonify({
        "boards": boards,
        "popular_counts": popular_counts,
        "unpopular_counts": unpopular_counts,
        "threshold": threshold
    })

@app.route('/get_4chan_hourly_heatmap', methods=['GET'])
def get_4chan_hourly_heatmap():
    start, end = parse_dates()
    days = (end - start).days + 1
    hourly_activity = [[0 for _ in range(24)] for _ in range(days)]

    cursor = g_tv_collection.find({
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        op_created_at = datetime.strptime(thread['original_post']['OP_Created_at'], '%Y-%m-%d %H:%M:%S')
        day_index = (op_created_at.date() - start.date()).days
        if 0 <= day_index < days:
            hourly_activity[day_index][op_created_at.hour] += 1
        for reply in thread.get("replies", []):
            reply_created_at = datetime.strptime(reply['Reply_Created_at'], '%Y-%m-%d %H:%M:%S')
            day_index = (reply_created_at.date() - start.date()).days
            if 0 <= day_index < days:
                hourly_activity[day_index][reply_created_at.hour] += 1

    day_labels = [(start + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(days)]
    return jsonify({
        "days": day_labels,
        "hours": list(range(24)),
        "matrix": hourly_activity
    })

@app.route('/get_4chan_popularity_histogram', methods=['GET'])
def get_4chan_popularity_histogram():
    board = request.args.get('board', 'g')
    start, end = parse_dates()
    reply_counts = []

    if board == 'pol':
        collection = pol_collection
    else:
        collection = g_tv_collection

    cursor = collection.find({
        "board": board,
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        reply_counts.append(len(thread.get("replies", [])))

    if not reply_counts:
        return jsonify({"labels": [], "counts": []})

    max_replies = max(reply_counts)
    bin_size = 10
    bins = list(range(0, max_replies + bin_size, bin_size))
    hist = [0]*(len(bins)-1)

    for rc in reply_counts:
        idx = rc // bin_size
        if idx >= len(hist):
            idx = len(hist)-1
        hist[idx] += 1

    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    return jsonify({
        "labels": labels,
        "counts": hist
    })

@app.route('/get_4chan_replies_vs_posts', methods=['GET'])
def get_4chan_replies_vs_posts():
    start, end = parse_dates()
    board_data = defaultdict(lambda: {"original_posts": 0, "replies": 0})

    cursor = g_tv_collection.find({
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end+timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        board = thread.get("board", "unknown")
        board_data[board]["original_posts"] += 1
        board_data[board]["replies"] += len(thread.get("replies", []))

    boards = sorted(board_data.keys())
    original_posts = [board_data[b]["original_posts"] for b in boards]
    replies = [board_data[b]["replies"] for b in boards]

    return jsonify({
        "boards": boards,
        "original_posts": original_posts,
        "replies": replies
    })

@app.route('/get_4chan_daily_data', methods=['GET'])
def get_4chan_daily_data():
    board = request.args.get('board', 'pol').strip()

    start_date_str = request.args.get('start_date', '2024-11-01')
    end_date_str = request.args.get('end_date', '2024-11-14')

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    if board == 'pol':
        collection = pol_collection
        board_filter = {"board": "pol"}
    elif board in ['g', 'tv']:
        collection = g_tv_collection
        board_filter = {"board": board}
    else:
        return jsonify({"error": "Invalid board."}), 400

    daily_counts = defaultdict(lambda: {"original_posts": 0, "replies": 0, "total_posts": 0})

    cursor = collection.find({
        **board_filter,
        "original_post.OP_Created_at": {
            "$gte": start_date.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": end_date.strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        op_created_at_str = thread['original_post']['OP_Created_at']
        op_created_at = datetime.strptime(op_created_at_str, '%Y-%m-%d %H:%M:%S')
        day_key = op_created_at.date()
        daily_counts[day_key]["original_posts"] += 1
        daily_counts[day_key]["total_posts"] += 1

        for reply in thread.get("replies", []):
            reply_created_at_str = reply['Reply_Created_at']
            reply_created_at = datetime.strptime(reply_created_at_str, '%Y-%m-%d %H:%M:%S')
            reply_day_key = reply_created_at.date()
            daily_counts[reply_day_key]["replies"] += 1
            daily_counts[reply_day_key]["total_posts"] += 1

    days = sorted(daily_counts.keys())
    labels = [day.strftime('%Y-%m-%d') for day in days]
    data = [daily_counts[day]["total_posts"] for day in days]

    return jsonify({"labels": labels, "data": data})

@app.route('/get_4chan_hourly_data', methods=['GET'])
def get_4chan_hourly_data():
    board = request.args.get('board', 'pol').strip()

    start_date_str = request.args.get('start_date', '2024-11-01')
    end_date_str = request.args.get('end_date', '2024-11-14')

    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')

    if board == 'pol':
        collection = pol_collection
        board_filter = {"board": "pol"}
    elif board in ['g', 'tv']:
        collection = g_tv_collection
        board_filter = {"board": board}
    else:
        return jsonify({"error": "Invalid board."}), 400

    hourly_counts = defaultdict(lambda: {"total_posts": 0})

    cursor = collection.find({
        **board_filter,
        "original_post.OP_Created_at": {
            "$gte": start_date.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": end_date.strftime('%Y-%m-%d %H:%M:%S')
        }
    })

    for thread in cursor:
        op_created_at_str = thread['original_post']['OP_Created_at']
        op_created_at = datetime.strptime(op_created_at_str, '%Y-%m-%d %H:%M:%S')
        hour_key = op_created_at.replace(minute=0, second=0)
        hourly_counts[hour_key]["total_posts"] += 1

        for reply in thread.get("replies", []):
            reply_created_at_str = reply['Reply_Created_at']
            reply_created_at = datetime.strptime(reply_created_at_str, '%Y-%m-%d %H:%M:%S')
            reply_hour_key = reply_created_at.replace(minute=0, second=0)
            hourly_counts[reply_hour_key]["total_posts"] += 1

    hours_sorted = sorted(hourly_counts.keys())
    labels = [h.strftime('%Y-%m-%d %H:%M') for h in hours_sorted]
    data = [hourly_counts[h]["total_posts"] for h in hours_sorted]

    return jsonify({"labels": labels, "data": data})

# ---------------------------Reddit Endpoints--------------------------------

def get_comments_per_hour(start_date, end_date, collection):
    logging.info(f"Fetching comments data from {start_date} to {end_date} for collection {collection.name}")

    hourly_counts = {date: {hour: 0 for hour in range(24)} for date in
                     (start_date + timedelta(n) for n in range((end_date - start_date).days + 1))}

    posts = collection.find({
        "crawled_at": {
            "$gte": start_date,
            "$lt": end_date + timedelta(days=1)
        }
    })

    for post in posts:
        crawled_at = post.get('crawled_at')
        if isinstance(crawled_at, datetime):
            hour = crawled_at.hour
            date = crawled_at.date()

            if date not in hourly_counts:
                hourly_counts[date] = {hour: 0 for hour in range(24)}

            hourly_counts[date][hour] += post.get('comment_count', 0)
        else:
            logging.warning(f"Invalid crawled_at value in post: {post}")

    return hourly_counts

@app.route('/get_comments_data', methods=['GET'])
def get_comments_data():
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        subreddit = request.args.get('subreddit', 'politics')
        
        start_date = datetime.strptime(start_date_str, "%Y-%m-%d") if start_date_str else datetime(2024, 11, 1)
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d") if end_date_str else datetime(2024, 11, 14)
        
        collection = db[f'reddit_{subreddit}']
        hourly_counts = get_comments_per_hour(start_date, end_date, collection)

        total_comments = 0
        peak_comments = 0
        peak_hour = ""
        for date, counts in hourly_counts.items():
            for hour, count in counts.items():
                total_comments += count
                if count > peak_comments:
                    peak_comments = count
                    peak_hour = f"{date} {hour}:00"

        total_deleted_posts = collection.count_documents({"is_deleted": True})
        active_users = len(collection.distinct("user_id"))

        total_hours = (end_date - start_date).days * 24
        avg_comments_hour = total_comments / total_hours if total_hours > 0 else 0

        chart_data = {
            "labels": [],
            "data": [],
            "total_comments": total_comments,
            "peak_comments": peak_comments,
            "peak_hour": peak_hour,
            "total_deleted_posts": total_deleted_posts,
            "active_users": active_users,
            "avg_comments_hour": round(avg_comments_hour, 2)
        }

        for date, counts in hourly_counts.items():
            for hour, count in counts.items():
                chart_data["labels"].append(f"{date} {hour}:00")
                chart_data["data"].append(count)

        return jsonify(chart_data)

    except Exception as e:
        logging.error(f"Error in get_comments_data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_reddit_submissions', methods=['GET'])
def get_reddit_submissions():
    try:
        start_date_str = request.args.get('start_date')
        end_date_str = request.args.get('end_date')
        subreddit = request.args.get('subreddit', 'politics')  

        if not start_date_str or not end_date_str:
            return jsonify({"error": "Missing start_date or end_date parameter"}), 400

        start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_date_str, "%Y-%m-%d")

        if start_date > end_date:
            return jsonify({"error": "start_date cannot be after end_date"}), 400

        collection = db[f'reddit_{subreddit}']
        daily_counts = {}

        posts = collection.find({
            "submitted_at": {
                "$gte": start_date,
                "$lt": end_date + timedelta(days=1)
            }
        })

        for post in posts:
            submitted_at = post.get('submitted_at')
            if isinstance(submitted_at, str):
                submitted_at = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))

            if isinstance(submitted_at, datetime):
                date = submitted_at.date()
                daily_counts[date] = daily_counts.get(date, 0) + 1
            else:
                logging.warning(f"Invalid submitted_at value: {post}")

        response = {
            "labels": [str(date) for date in sorted(daily_counts.keys())],
            "data": [daily_counts[date] for date in sorted(daily_counts.keys())]
        }

        return jsonify(response)
    except Exception as e:
        logging.error(f"Error fetching submissions data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_deleted_posts', methods=['GET'])
def get_deleted_posts():
    try:
        subreddit = request.args.get('subreddit', 'politics')  
        start_date = datetime(2024, 11, 1)
        end_date = datetime(2024, 11, 15)

        collection = db[f'reddit_{subreddit}']
        deleted_counts = {}

        posts = collection.find({
            "submitted_at": {
                "$gte": start_date,
                "$lt": end_date + timedelta(days=1)
            }
        })

        for post in posts:
            submitted_at = post.get('submitted_at')
            if isinstance(submitted_at, str):
                submitted_at = datetime.fromisoformat(submitted_at.replace('Z', '+00:00'))
            if isinstance(submitted_at, datetime):
                date = submitted_at.date()
                is_deleted = post.get('is_deleted', False)  
                if is_deleted:
                    deleted_counts[date] = deleted_counts.get(date, 0) + 1

        response = {
            "labels": [str(date) for date in sorted(deleted_counts.keys())],
            "data": [deleted_counts[date] for date in sorted(deleted_counts.keys())]
        }

        return jsonify(response)
    except Exception as e:
        logging.error(f"Error fetching deleted posts data: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_comment_upvotes', methods=['GET'])
def get_comment_upvotes_endpoint():
    try:
        subreddit = request.args.get('subreddit', 'technology')
        start_date = request.args.get('start_date', '2024-11-21')
        end_date = request.args.get('end_date', '2024-11-28')

        start_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

        collection = collections.get(subreddit)
        if collection is None:
            return jsonify({"error": f"Invalid subreddit: {subreddit}"}), 400

        pipeline = [
            {"$match": {"subreddit": subreddit, "crawl_history": {"$gte": start_date, "$lte": end_date}}},
            {"$unwind": "$comments"},
            {"$group": {
                "_id": "$post_id",
                "comment_upvotes": {"$push": "$comments.upvote_score"},
                "post_title": {"$first": "$post_title"}
            }}
        ]
        result = list(collection.aggregate(pipeline))
        return jsonify(result)

    except Exception as e:
        logging.error(f"Error in get_comment_upvotes_endpoint: {e}", exc_info=True)
        return jsonify({"error": str(e)}), 500

@app.route('/ask_question', methods=['POST'])
def ask_question():
    user_prompt = request.form.get('prompt', '').strip()
    subreddit = request.form.get('subreddit', 'politics').strip()

    valid_subreddits = ['technology', 'movies', 'politics']
    if not user_prompt:
        return jsonify({"error": "No prompt provided. Please include a valid question."}), 400

    if subreddit not in valid_subreddits:
        return jsonify({"error": f"Invalid subreddit. Choose from {', '.join(valid_subreddits)}."}), 400

    try:
        if subreddit == 'politics':
            collection_name = 'reddit_politics'
            query = {}
        else:
            collection_name = 'posts'
            query = {
                'subreddit': subreddit,
                'post_title': {'$regex': '|'.join(user_prompt.split()), '$options': 'i'}
            }

        if collection_name not in db.list_collection_names():
            return jsonify({"error": f"No data available for subreddit: {subreddit}."}), 404

        collection = db[collection_name]
        recent_posts = list(collection.find(query).limit(10))

        if not recent_posts:
            return jsonify({"error": f"No relevant posts found for your query in subreddit: {subreddit}."}), 404

        formatted_data = "\n".join([f"- {post.get('post_title', 'No Title')}" for post in recent_posts])
        llm_input = f"""
        User Prompt:
        {user_prompt}

        Relevant Data from Subreddit '{subreddit}':
        {formatted_data}

        Provide a detailed and coherent response based on the above data.
        """

        input_ids = tokenizer.encode(llm_input, return_tensors="pt")

        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        attention_mask = input_ids.ne(tokenizer.pad_token_id).long()

        output = model.generate(
            input_ids,
            max_length=500,
            temperature=0.6,
            top_p=0.9,
            do_sample=True,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id
        )
        answer = tokenizer.decode(output[0], skip_special_tokens=True)

        answer_lines = answer.split('. ')
        formatted_answer = "\n\n".join([f"{i + 1}. {line.strip()}" for i, line in enumerate(answer_lines) if line.strip()])

        return jsonify({"answer": formatted_answer})

    except Exception as e:
        return jsonify({"error": f"Server error occurred: {str(e)}"}), 500

# ---------------------------New Toxicity Endpoint for 4chan----------------------------

@app.route('/get_4chan_toxicity_data', methods=['GET'])
def get_4chan_toxicity_data():
    board = request.args.get('board', 'g')
    start, end = parse_dates(toxicity=True)
    is_deleted_param = request.args.get('is_deleted', '')
    
    is_deleted_filter = {}
    if is_deleted_param == 'true':
        is_deleted_filter["is_deleted"] = True
    elif is_deleted_param == 'false':
        is_deleted_filter["is_deleted"] = False

    cursor = mod_coll.find({
        "board": board,
        "original_post.OP_Created_at": {
            "$gte": start.strftime('%Y-%m-%d %H:%M:%S'),
            "$lte": (end + timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S')
        },
        **is_deleted_filter
    })

    daily_toxicity = defaultdict(lambda: defaultdict(int))

    for thread in cursor:
        op_created_at = datetime.strptime(thread['original_post']['OP_Created_at'], '%Y-%m-%d %H:%M:%S')
        day_key = op_created_at.date()

        op_class = thread.get("original_post_toxicity", {}).get("class", "unknown")
        daily_toxicity[day_key][op_class] += 1

        for reply_tox in thread.get("replies_toxicity", []):
            rclass = reply_tox.get("class", "unknown")
            daily_toxicity[day_key][rclass] += 1

    days = sorted(daily_toxicity.keys())
    labels = [d.strftime('%Y-%m-%d') for d in days]

    all_classes = set()
    for day in daily_toxicity:
        for cls in daily_toxicity[day]:
            all_classes.add(cls)
    all_classes = sorted(all_classes)

    counts = {cls: [] for cls in all_classes}
    for d in days:
        for cls in all_classes:
            counts[cls].append(daily_toxicity[d][cls])

    return jsonify({
        "labels": labels,
        "classes": list(all_classes),
        "counts": counts
    })


if __name__ == '__main__':
    logging.info("Starting Flask application")
    app.run(debug=True)
