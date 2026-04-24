"""10 Multi-Turn Test Conversations for benchmarking memory vs no-memory agents.

Each conversation is a list of user messages that test different memory capabilities.
"""

CONVERSATIONS: list[dict] = [
    {
        "id": "conv_01_preferences",
        "name": "Personal Preferences Recall",
        "description": "Tests if the agent remembers user preferences across turns",
        "turns": [
            "My name is Khanh and I'm a software engineer",
            "I prefer Python over JavaScript for backend development",
            "I always use dark mode in my editors",
            "What's my name and what language do I prefer for backend?",
            "Set up a new project for me — which language and theme should you use?",
        ],
    },
    {
        "id": "conv_02_continuity",
        "name": "Multi-Session Continuity",
        "description": "Tests if context carries over across multiple turns",
        "turns": [
            "I'm working on a machine learning project for image classification",
            "I've decided to use PyTorch for the framework",
            "The dataset has 10,000 images across 5 classes",
            "What framework am I using and how many classes?",
            "Suggest a model architecture based on what you know about my project",
        ],
    },
    {
        "id": "conv_03_technical_qa",
        "name": "Technical Q&A with Context",
        "description": "Tests factual recall within a technical conversation",
        "turns": [
            "Explain what a transformer architecture is",
            "How does self-attention work in transformers?",
            "What is the difference between encoder and decoder?",
            "Summarize the three concepts we just discussed",
            "How do they relate to BERT and GPT models?",
        ],
    },
    {
        "id": "conv_04_task_planning",
        "name": "Task Planning with History",
        "description": "Tests memory-aided task planning",
        "turns": [
            "I need to build a REST API. Step 1: Design the database schema",
            "The API will handle users, posts, and comments",
            "Step 2: Set up the project with FastAPI and PostgreSQL",
            "What was step 1 and what entities are we working with?",
            "Create step 3 based on what we've planned so far",
        ],
    },
    {
        "id": "conv_05_emotional",
        "name": "Emotional Context Tracking",
        "description": "Tests if the agent picks up on and remembers emotional context",
        "turns": [
            "I'm really frustrated with this bug I've been debugging for 3 hours",
            "It turns out it was just a missing semicolon...",
            "I feel much better now. Thanks for listening!",
            "How was I feeling at the start of our conversation?",
            "Give me a motivational message based on our conversation",
        ],
    },
    {
        "id": "conv_06_code_debug",
        "name": "Code Debugging Across Turns",
        "description": "Tests memory of code context and debugging progression",
        "turns": [
            "I have a Python function that calculates fibonacci but it's too slow for n=50",
            "I'm currently using recursive approach without memoization",
            "Can you suggest an optimized version?",
            "What was the original problem with my fibonacci function?",
            "Compare the time complexity of my original vs your suggested approach",
        ],
    },
    {
        "id": "conv_07_learning",
        "name": "Learning Progression Tracking",
        "description": "Tests if the agent tracks learning progress",
        "turns": [
            "I'm a beginner learning Docker. I understand what containers are",
            "Now teach me about Docker Compose",
            "I've mastered Compose. What about Docker networking?",
            "Summarize my Docker learning journey so far",
            "What should I learn next based on my progression?",
        ],
    },
    {
        "id": "conv_08_recommendations",
        "name": "Recommendation Refinement",
        "description": "Tests progressive refinement using remembered preferences",
        "turns": [
            "Recommend me a programming book. I like practical, hands-on books",
            "I've already read Clean Code and Design Patterns",
            "I prefer books focused on Python or system design",
            "Based on everything you know about my preferences, give me your top 3",
            "Why did you choose those specific books for me?",
        ],
    },
    {
        "id": "conv_09_reasoning",
        "name": "Complex Reasoning Chains",
        "description": "Tests if memory helps maintain reasoning chain",
        "turns": [
            "Let's analyze a system design: We need a chat application for 1M users",
            "The requirements are: real-time messaging, message history, user presence",
            "We chose WebSocket for real-time and Cassandra for message storage",
            "What were our original requirements and what technologies did we choose?",
            "Identify potential bottlenecks given our choices and scale",
        ],
    },
    {
        "id": "conv_10_mixed_intent",
        "name": "Mixed-Intent Conversations",
        "description": "Tests memory router with different intents in same conversation",
        "turns": [
            "Remember that my timezone is GMT+7 and I work from 9am to 6pm",
            "What is the CAP theorem in distributed systems?",
            "Find topics similar to distributed systems that we've discussed",
            "What timezone am I in? And summarize what we talked about today",
            "Based on our past conversations, what topics interest me most?",
        ],
    },
]
