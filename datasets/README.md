# Cascade data

Extract `cascades.zip` here. Each dataset is one Zstandard-compressed Parquet file at `cascades/<dataset>.parquet`; each row is a complete upstream cascade.

| Field | Meaning |
| --- | --- |
| `cascade_index` | Zero-based source row position |
| `cascade_id` | Original cascade ID, as text |
| `root_id` | Original root node ID, including virtual roots |
| `published` | Original publication timestamp, date or year, as text |
| `reported_size` | Upstream size declaration, not a label or event count |
| `events` | Ordered list of `{path: list<int64>, elapsed: int64}` records |

Arrival times use seconds for Weibo/Twitter, days for APS, and years for ACM/DBLP. The file's `cascade_metadata` records its dataset, units, format version and upstream SHA-256. ACM includes 711 original `published="0"` values, retained as supplied.

Paths, source order, repeated adopters, negative times and virtual roots are preserved. Preparation applies the model's filtering, stable time sort, splits, labels and 100-event input limit, then saves `.pt` tensors. Each upstream event is a path/time record, not necessarily a distinct user.
