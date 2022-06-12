# ECE276B PR1 README

To run the code, simply run

```bash
python3 main.py --part A
python3 main.py --part B
```

Argument part controls to run part A or part B. Running partA will generate 8 gifs (including example.env) in the folder gifs and partB will generate 36 gifs in foler rd_gifs.

The file structure is as follow

- main.py - runner entrance for the whole program
- utils.py - basic utils for the program and DP algorithm
- dynamic_programming.py - algorithm class that solves partA
- dp_random_map.py - algorithm class that solves partB