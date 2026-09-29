Median of 2 interleaved repetitions; min–max across repetitions in parentheses.

| Policy | Concurrency | Req/s | p50 ms | p99 ms | Server-side mean ms | Texts per model call | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| default server, httpx client | 16 | 200 (191–210) | 75.0 (72.3–77.7) | 157.8 (137.9–177.7) | 61.4 | 6.8 | 0 |
| default server, httpx client | 64 | 81 (79–82) | 517.6 (494.9–540.3) | 3146.4 (2993.6–3299.3) | 64.5 | 1.4 | 0 |
| default server, aiohttp client | 16 | 206 (203–209) | 74.0 (72.7–75.2) | 129.0 (125.3–132.6) | 70.2 | 7.9 | 0 |
| default server, aiohttp client | 64 | 194 (192–197) | 317.0 (316.5–317.5) | 481.5 (460.0–502.9) | 296.5 | 22.5 | 0 |
