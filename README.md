## Master's Thesis Repository

This repository supports my Master's thesis, which investigates the following research question:

> **What patterns of literary influence and social connectivity emerge from the borrowing activity among author-members of the Shakespeare and Company lending library in the early 20th century?**

### Definition of Author-Members

**Author-members** are defined as individuals who had at least one book—authored solely by themselves—available in the Shakespeare and Company lending library.

### Repository Contents

This repository contains a series of Jupyter notebooks and supplementary analyses that explore various dimensions of the borrowing activity. The notebooks are organized into the following thematic sections:

1. **Calculation of Author-Member Intersections**  
   - Identifies shared characteristics among author-members based on their borrowing activity.
   - The resulting dataset (`01_author_is_member.csv`) is used as the foundation for subsequent analyses.

2. **Geographical Analysis of Author-Members**  
   - **By Region:** Investigates the global distribution of author-members.  
   - **In and Around Paris:** Focuses on the spatial distribution within Paris and its environs.  
   - **Shared Addresses:** Explores instances of shared addresses among author-members to infer possible social connections.

3. **Statistical Analysis**  
   - **Nationalities:** Analyzes the national origins of the author-members.  
   - **Books:** Quantifies the contributions of author-members to the lending library's catalog.

4. **Network Analysis of Borrowing Activity**  
   - **All Books:**  
     - Examines the overall borrowing activity of any title among author-members in the library.  
     - Includes computations of network metrics such as centrality and clustering.  
   - **Author-Member Books Only:**  
     - Focuses exclusively on borrowing patterns among author-members for books authored by the author-members.  
     - Provides network metrics specific to this subset.
   - Additionally, bipartite graphs for both subgroups are created to visualize the borrowing activity among author-members by including borrowed titles.

These analyses collectively aim to illuminate the cultural and social dynamics that characterized the lending practices and interpersonal relationships of early twentieth-century literary communities.

### Data Sources

The data analyzed in this project were obtained from the [Shakespeare and Company Project](https://shakespeareandco.princeton.edu/), a digital humanities initiative dedicated to reconstructing the lending library's collection and membership. Detailed metadata and documentation are provided within the repository.

### Requirements and Environment Setup

The analyses were conducted using Python 3.7. All necessary dependencies are listed in the `requirements.txt` file. To replicate the environment, execute the following command in your terminal:

```bash
pip install -r requirements.txt
```

For enhanced reproducibility, consider using a virtual environment (e.g., `venv` or `conda`).

### File Structure Overview
Below is an overview of the repository’s file structure.

```plaintext
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # Contains the original data files as obtained from external sources.
│   └──  processed/          # Includes data files that have been cleaned, filtered, or otherwise transformed.
├── notebooks/
|   ├── 00_introduction.ipynb        # Notebook providing an overview of the research question and methodology.
│   ├── 01_intersection.ipynb    # Notebook determining the intersection of authors and members.
│   ├── 02_geospatial_analysis.ipynb # Notebook focused on geographical analysis of author-members.
│   ├── 03_statistical_analysis.ipynb# Notebook containing statistical examinations of nationalities and book contributions.
│   └── 04_network_analysis.ipynb    # Notebook dedicated to network analysis of borrowing activity.
├── utils.py              # Contains reusable functions (in this case, name sorting) referenced across notebooks.
└── (Other configuration files, such as .gitignore, etc.)
```

### Explanation of Key Components

**README.md**  
Provides an overview of the project, including the research question, methodology, and instructions for environment setup and execution of the notebooks.

**requirements.txt**  
Lists the Python packages required to replicate the analysis. This ensures that users can recreate the computational environment by installing the specified versions of libraries.

**data/**  
The data directory is subdivided into:
- **raw/**: Contains the original, unmodified datasets.
- **processed/**: Houses datasets that have undergone cleaning or filtering (e.g., the CSV file with filtered entries).

**notebooks/**  
Contains the Jupyter notebooks that form the core of the exploratory data analysis. Each notebook is dedicated to a specific aspect of the analysis—from data acquisition through various analytical approaches (geographical, statistical, network-based).

**utils.py**  
A Python module with helper functions reused across multiple notebooks. This modular approach enhances maintainability and ensures consistency in common tasks.

### Usage and Execution

Each Jupyter notebook in this repository is self-contained and includes explanatory markdown cells detailing the methodology, results, and interpretations. Users are encouraged to review these narrative sections to fully understand the analytical approach and outcomes. It is highly advised to read the notebooks in the order of their numerical prefixes (e.g., `00_introduction.ipynb`, `01_intersection.ipynb`, etc.) to follow the logical progression of the analysis.

### Acknowledgments

This research is part of my Master's thesis in Digital Humanities. I express my gratitude to my supervisor and the academic community for their guidance and insights.
