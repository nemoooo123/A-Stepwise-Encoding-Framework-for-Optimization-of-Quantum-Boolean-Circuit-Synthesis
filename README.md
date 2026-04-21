# A Stepwise Encoding Framework for Optimization of Quantum Boolean Circuit Synthesis

---

## 🚀 Introduction

Quantum circuit synthesis is one of the key problems for making quantum computing practical, but it is also very difficult to solve. Existing mainstream methods can generally be divided into two types: heuristic methods and metaheuristic methods. Heuristic methods have better scalability and can handle circuit synthesis problems with a larger number of qubits, but the quality of the synthesized circuits is usually worse. Metaheuristic methods may produce higher-quality circuits, but they are often limited to smaller problem sizes. Therefore, how to achieve both large synthesis scale and high circuit quality remains an important challenge in quantum circuit synthesis.

This study focuses on quantum Boolean circuit synthesis and proposes a synthesis framework that combines the advantages of heuristic and metaheuristic methods. The core idea is to gradually transform a heuristic synthesis process, which preserves multiple synthesis possibilities, into a specific encoding representation, so that the resulting search space corresponds to the feasible solution set. Since this feasible solution set still grows exponentially and has the properties of a high-dimensional discrete space, it is well suited for metaheuristic optimization. In this study, the Amplitude-Ensemble Quantum-Inspired Tabu Search Algorithm (AE-QTS) is mainly used to search for high-quality solutions in this space, in order to demonstrate the effectiveness of the encoding. AE-QTS is inspired by quantum algorithms, is very easy to implement, and shows strong and stable performance, making it suitable for efficiently finding high-quality solutions in a high-dimensional discrete feasible solution space.

The experimental results show that the proposed method greatly increases the solvable synthesis scale from 4 qubits, which is typical for traditional metaheuristic methods, to 13 qubits, while still maintaining good circuit quality. In addition, the analysis results on all 3-qubit test instances show that the quality distribution of the solutions obtained by the proposed method is very close to that of the optimal solutions. These results indicate that this study provides a feasible synthesis approach with both scalability and optimization capability. It also has strong potential for further development and may open up a completely new branch of optimization for quantum Boolean circuit synthesis.

---


## 📖 Problem Representation
To test new problems, please refer to the following format:

### 1. Truth Table Format
This program represents quantum circuit problems as Permutation Truth Tables. An $n$-bit problem must be represented as an integer list of length $2^n$, containing a unique permutation of values from $0$ to $2^n-1$.

* **Example (3-bit problem)**: `[1, 0, 3, 2, 5, 7, 4, 6]`
* **Mapping**:
    * 0 (000) &rarr; 1 (001)
    * 1 (001) &rarr; 0 (000)
    * 7 (111) &rarr; 6 (110)

### 2. Benchmark Library

The project includes several built-in benchmarks for quantum Boolean circuit synthesis, ranging from 3-bit to 13-bit truth tables.

| Qubits (Bits) | Problems | Qubits (Bits) | Problems |
| :--- | :--- | :--- | :--- |
| **3-bit** | 13 | **9-bit**  | 5 |
| **4-bit** | 9  | **10-bit** | 5 |
| **5-bit** | 5  | **11-bit** | 5 |
| **6-bit** | 5  | **12-bit** | 5 |
| **7-bit** | 5  | **13-bit** | 5 |
| **8-bit** | 5  |

---

## 詳細題目資料 (Truth Tables)

<details>
<summary>▶ 3-Bit 基準函數 (f1 - f9)</summary>

### 3 Bits ($2^3 = 8$)
* **f1:** `[1, 0, 3, 2, 5, 7, 4, 6]`
* **f2:** `[7, 0, 1, 2, 3, 4, 5, 6]`
* **f3:** `[0, 1, 2, 3, 4, 6, 5, 7]`
* **f4:** `[0, 1, 2, 4, 3, 5, 6, 7]`
* **f5:** `[1, 2, 3, 4, 5, 6, 7, 0]`
* **f6:** `[3, 6, 2, 5, 7, 1, 0, 4]`
* **f7:** `[1, 2, 7, 5, 6, 3, 0, 4]`
* **f8:** `[4, 3, 0, 2, 7, 5, 6, 1]`
* **f9:** `[7, 5, 2, 4, 6, 1, 0, 3]`
</details>

<details>
<summary>▶ 4-Bit 基準函數 (f10 - f16)</summary>

### 4 Bits ($2^4 = 16$)
* **f10:** `[0, 1, 14, 15, 4, 5, 10, 11, 7, 9, 6, 8, 12, 13, 2, 3]`
* **f11:** `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0]`
* **f12:** `[0, 7, 6, 9, 4, 11, 10, 13, 8, 15, 14, 1, 12, 3, 2, 5]`
* **f13:** `[6, 2, 14, 13, 3, 11, 10, 7, 0, 5, 8, 1, 15, 12, 4, 9]`
* **f14:** `[0, 9, 10, 5, 4, 15, 14, 8, 11, 2, 6, 3, 12, 7, 1, 13]`
* **f15:** `[6, 4, 11, 0, 9, 8, 12, 2, 15, 5, 3, 7, 10, 13, 14, 1]`
* **f16:** `[13, 1, 14, 0, 9, 2, 15, 6, 12, 8, 11, 3, 4, 5, 7, 10]`
</details>

<details>
<summary>▶ 5-Bit 基準函數 (f17 - f21)</summary>

### 5 Bits ($2^5 = 32$)
* **f17:** `[21, 6, 3, 0, 9, 26, 15, 12, 13, 14, 27, 8, 1, 2, 7, 20, 5, 22, 18, 16, 25, 10, 31, 28, 29, 30, 11, 24, 17, 19, 23, 4]`
* **f18:** `[0, 16, 1, 29, 30, 22, 10, 17, 15, 7, 18, 13, 3, 21, 25, 11, 8, 4, 20, 14, 5, 9, 26, 12, 6, 31, 2, 23, 19, 27, 24, 28]`
* **f19:** `[0, 1, 2, 3, 4, 5, 6, 27, 7, 8, 9, 28, 10, 29, 30, 31, 11, 12, 13, 16, 14, 17, 18, 19, 15, 20, 21, 22, 23, 24, 25, 26]`
* **f20:** `[16, 17, 18, 3, 19, 4, 5, 20, 21, 6, 7, 22, 8, 23, 24, 9, 25, 10, 11, 26, 12, 27, 28, 13, 14, 29, 30, 15, 31, 0, 1, 2]`
* **f21:** `[16, 17, 18, 19, 0, 20, 21, 22, 23, 24, 25, 11, 12, 26, 27, 15, 28, 13, 14, 29, 8, 9, 10, 30, 31, 1, 2, 3, 4, 5, 6, 7]`
</details>

<details>
<summary>▶ 6-Bit 基準函數 (f22 - f26)</summary>

### 6 Bits ($2^6 = 64$)
* **f22:** `[在此處貼上你的資料]`
* **f23:** `[...]`
* **f24:** `[...]`
* **f25:** `[...]`
* **f26:** `[...]`
</details>

---



### 3. Execution and Customization
All problem data is managed by `utils/data_loader.py` .

#### Running Built-in Problems:

Specify `num_bits` and `problem_idx` according to the table above when executing the program.

* Example: 
To run the 1st problem for 6-bit circuits, use the parameters: 6 1 [algo_id]。

#### Adding Custom Problems:

1. Open `utils/data_loader.py`.

2. Define your truth table within the corresponding bit-width dictionary in the `DataLoader` class.

3. Save the file and execute using the command line parameters.

---

## 💻 Usage

### Requirements

Python: v3.10 or higher (Tested on v3.13).


Dependencies:

```Bash
pip install numpy matplotlib pandas
```


### Execution Guide

 The program supports command-line arguments for batch processing or server-side experiments:

```Bash
python main.py [num_bits] [problem_idx] [algo_id]
```
Algorithm ID Mapping:

1: Amplitude-Ensemble Quantum-Inspired Tabu Search Algorithm (AE-QTS)

2: Quantum-Inspired Tabu Search (QTS)

3: Quantum Evolutionary Algorithm (QEA)

4: Genetic Algorithm (GA)

5: Differential Evolution (DE)

6: Tubu Search (TS)

7: Particle Swarm Optimization (PSO)

8: Whale Optimization Algorithm (WOA)

9: Artificial Bee Colony Algorithm (ABC)

Example Command:

```Bash
# Run 3-bit Problem #1 using the AE-QTS algorithm
python main.py 3 1 1
```

## 📂 Experimental Results
Upon execution, the system automatically creates a structured directory under exp/ to store comprehensive trial data. Each experiment generates two types of files: a Summary Report (.txt) for quick review and a Convergence Matrix (.xlsx) for deep statistical analysis.

Path Format:
```Bash
exp/
└── [n]_bit/                         # Target qubit count (e.g., 13_bit)
    └── [Algorithm]_Results/         # Specific algorithm folder
        ├── [Algorithm]_[n]_[idx].txt    # Summary, Best Circuit, & Statistics
        └── [Algorithm]_[n]_[idx].xlsx   # Full Generation-by-Generation Data
```

| File Type | Content Description | Purpose |
| :--- | :--- | :--- | 
| **.txt** | Global best gate count, optimal circuit structure, average execution time, and final μ±σ results. | Quick Verification: Rapidly check if the algorithm found the target gate count.|
| **.xlsx** | A complete Trial×Generation fitness matrix, including row-wise averages and standard deviations for every step.  | Data Analysis: Used for plotting convergence curves and performing T-tests. |


## 📊 Data Analysis
This project includes automated plotting scripts that read experimental data from the ```exp/``` folder and generate high-quality PDF vector graphics for publications:

```Bash
# Run the plotting script
python plot_results.py
```
Generated charts are saved in the ```Comparison_PDF_Plots/``` directory.

### Adding New Problems to Plots

If you add new truth tables or bit-widths, update the `tasks` dictionary in `plot_results.py` to ensure the script retrieves the data:

1. Open `plot_results.py`。
2. Locate the `tasks` dictionary :
   ```python
   tasks = {
    3: [1, 13], # 3-bit: Problems 1 to 13
    4: [1, 9],  # 4-bit: Problems 1 to 9
    # Add new bit-widths or extend indices here
    8: [1, 5] # Example: 8-bit Problems 1 to 5 
    }  

## 📜 Copyright & Authorship

This project was developed by Wei-Chieh Lai under the supervision of Prof. Kuo-Chun Tseng at Advanced System Circuit Lab (ASC Lab), National Ilan University.

* **Author:** Wei-Chieh Lai

* **Supervisor:** Prof. Kuo-Chun Tseng

* **Affiliation:** ASC Lab, Department of Computer Science and Information Engineering, National Ilan University

* **Copyright:** © 2026 Wei-Chieh Lai and Prof. Kuo-Chun Tseng, ASC Lab, National Ilan University.

* **License:** This project is licensed under the MIT License.
## 📬 Contact
For inquiries regarding algorithm implementation, problem formats, or academic collaboration, please contact:

* **Email:** R1443006@ems.niu.edu.tw

* **Lab:** Advanced System Circuit Lab (ASC Lab)

* **Affiliation:** National Ilan University, Taiwan