# Multi-Memory Agent with LangGraph

## Mô tả

Xây dựng Multi-Memory Agent sử dụng LangGraph với đầy đủ memory stack gồm 4 backend:

1. **ConversationBufferMemory** (Short-term) — Bộ nhớ ngắn hạn trong RAM
2. **Redis** (Long-term) — Lưu trữ dài hạn (user preferences, facts)
3. **JSON Episodic Log** — Ghi chép từng episode tương tác
4. **Chroma Semantic** — Tìm kiếm ngữ nghĩa bằng vector embeddings

## Kiến trúc

```
User Query → Memory Router → [Route to appropriate memory]
                                    ↓
              Context Window Manager (auto-trim, priority eviction)
                                    ↓
                        LangGraph Agent (OpenAI GPT-4o-mini)
                                    ↓
                          Memory Writer (store to all backends)
```

## Cài đặt

```bash
# 1. Clone repository
git clone <repo-url>
cd 2A202600200-NguyenQuocKhanh-Lab17

# 2. Tạo virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Cài đặt dependencies
pip install -r requirements.txt

# 4. Cấu hình API key
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Chạy Benchmark

```bash
python -m benchmark.run_benchmark
```

Benchmark sẽ:
- Chạy 10 multi-turn conversations (5 turns mỗi conversation)
- So sánh agent **có memory** vs **không có memory**
- Đo 3 metrics: Response Relevance, Context Utilization, Token Efficiency
- Xuất báo cáo ra `docs/benchmark_report.md`

## Chạy Tests

```bash
pytest tests/ -v
```

## Cấu trúc Project

```
├── src/
│   ├── agent.py              # LangGraph agent
│   ├── config.py             # Configuration
│   ├── context_manager.py    # Context window management
│   └── memory/
│       ├── base.py           # Abstract base class
│       ├── buffer_memory.py  # Short-term memory
│       ├── redis_memory.py   # Long-term Redis memory
│       ├── episodic_memory.py # JSON episodic log
│       ├── semantic_memory.py # Chroma semantic memory
│       └── router.py         # Memory router
├── benchmark/
│   ├── conversations.py      # 10 test conversations
│   ├── metrics.py           # Evaluation metrics
│   └── run_benchmark.py     # Benchmark runner
├── tests/
│   └── test_memory.py       # Unit tests
├── data/                    # Runtime data storage
├── docs/
│   └── benchmark_report.md  # Generated benchmark report
├── requirements.txt
└── .env.example
```

## Memory Router Logic

| Query Intent | Memory Backend | Example |
|---|---|---|
| User Preference | Redis (Long-term) | "What language do I prefer?" |
| Factual Recall | ConversationBuffer (Short-term) | "What did we just discuss?" |
| Experience Recall | JSON Episodic | "Last time we talked about..." |
| Semantic Search | Chroma (Semantic) | "Find similar topics to..." |

## Context Window Management

4-level priority eviction khi context gần đầy:

| Priority | Level | Content | Eviction |
|---|---|---|---|
| 0 | Critical | System prompt, current query | Never |
| 1 | High | Recent 3 turns | Last resort |
| 2 | Medium | Retrieved memories | When needed |
| 3 | Low | Old history, episodic logs | First to evict |

## Conflict Handling

Khi user đính chính một thông tin (ví dụ: dị ứng sữa bò → đậu nành), `RedisLongTermMemory` ghi đè entry cũ thay vì append:

```python
await redis_memory.store_preference("allergy", "dairy")   # lưu lần đầu
await redis_memory.store_preference("allergy", "soy")     # ghi đè → không còn "dairy"
```

Nguyên lý: preference dùng **stable Redis key** `ltm:pref:<key>` thay vì UUID ngẫu nhiên, đảm bảo mỗi category chỉ có 1 giá trị mới nhất.

## Reflection: Privacy & Limitations

Xem chi tiết tại [`docs/REFLECTION.md`](docs/REFLECTION.md).

Tóm tắt các điểm quan trọng:

| Rủi ro / Giới hạn | Mức độ |
|-------------------|--------|
| Long-term Redis lưu PII (tên, dị ứng, lịch) — không có consent | 🔴 Cao |
| Fakeredis không persist qua restart | 🔴 Cao |
| Không có user isolation (multi-tenant) | 🔴 Cao |
| Semantic search có thể retrieve false positive | 🟡 Trung bình |
| JSONL episodic không concurrent-safe | 🟡 Trung bình |
| LLM-based routing tốn thêm 1 API call/turn | 🟡 Trung bình |

