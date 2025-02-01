## Master's Thesis Repository

This repository supports my Master's thesis, which investigates the following research question:

> **What patterns of literary influence and social connectivity emerge from the borrowing activity among author-members of the Shakespeare and Company lending library in the early 20th century?**

### Definition of Author-Members

**Author-members** are defined as individuals who had at least one book—authored solely by themselves—available in the Shakespeare and Company lending library.

### Repository Contents

This repository contains a series of Jupyter notebooks and supplementary analyses that explore various dimensions of the borrowing activity. The notebooks are organized into the following thematic sections:

1. **Calculation of Author-Member Intersections**  
   - Identifies shared characteristics among author-members based on their borrowing activity.

2. **Geographical Analysis of Author-Members**  
   - **By Continent:** Investigates the global distribution of author-members.  
   - **In and Around Paris:** Focuses on the spatial distribution within Paris and its environs.  
   - **Shared Addresses:** Explores instances of shared addresses among author-members to infer possible social connections.

3. **Statistical Analysis**  
   - **Nationalities:** Analyzes the national origins of the author-members.  
   - **Books:** Quantifies the contributions of author-members to the lending library's catalog.

4. **Network Analysis of Borrowing Activity**  
   - **All Books:**  
     - Examines the overall borrowing activity in the library.  
     - Includes computations of network metrics such as centrality and clustering.  
   - **Author-Member Books Only:**  
     - Focuses exclusively on borrowing patterns for books authored by the author-members.  
     - Provides network metrics specific to this subset.

These analyses collectively aim to illuminate the cultural and social dynamics that characterized the lending practices and interpersonal relationships of early twentieth-century literary communities.

### Data Sources

The data analyzed in this project were obtained from the [Shakespeare and Company Project](https://shakespeareandco.princeton.edu/), a digital humanities initiative dedicated to reconstructing the lending library's collection and membership. Detailed metadata and documentation are provided within the repository.

### Requirements and Environment Setup

The analyses were conducted using Python 3.7. All necessary dependencies are listed in the `requirements.txt` file. To replicate the environment, execute the following command in your terminal:

```bash
pip install -r requirements.txt