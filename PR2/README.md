# ECE276B PR2 README

To run the code, simply run

```bash
python3 main.py
```

Implemented 3 kinds of search-based algorithms namely A* / RTAA* / JPS. To run different planner, plz uncomment corresponding planner on 47 / 48 / 49 lines of ```main.py```. Also note that each planner file is also runable for testing single phase planning instead of a moving target.

The file structure is as follow

- main.py - runner entrance for the whole program
- a_star.py - AStar Planner Implementation
- rtaa_star.py - RTAA* Planner Implementation
- jps.py - JPS Planner Implementation
- robotplanner.py - original greedy planner and another simple A*