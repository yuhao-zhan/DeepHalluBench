# DeepHalluBench
This is the official repository for the paper: **"Why Your Deep Research Agent Fails? On Hallucination Evalu ation in Full Research Trajectory"**.

![image-20260130215645833](assets/first_image.png)

## 📢 Release Schedule (Coming Soon)

We are currently finalizing the organization of the codebase and the **DeepHalluBench** dataset. Due to internal procedures, the data and code cannot be uploaded immediately.

**We plan to release the full data and code within one month.** Please **Star** ⭐️ and **Watch** 👀 this repository to get notified immediately upon release!

---

## Abstract

Diagnosing the failure mechanisms of Deep Research Agents (DRAs) remains a critical challenge. Existing benchmarks predominantly rely on end-to-end evaluation, obscuring critical intermediate hallucinations, such as flawed planning, that accumulate throughout the research trajectory. To bridge this gap, we propose a shift from outcome-based to **process-aware evaluation** by auditing the full research trajectory. We introduce the **PIES Taxonomy** to categorize hallucinations along functional components (<u>**P**</u>lanning vs. <u>**S**</u>ummarization) and error properties (<u>**E**</u>xplicit vs. <u>**I**</u>mplicit). We instantiate this taxonomy into a fine-grained evaluation framework that decomposes the trajectory to rigorously quantify these hallucinations. Leveraging this framework to isolate 100 distinctively hallucination-prone tasks including adversarial scenarios, we curate **DeepHalluBench**. Experiments on six state-of-theart DRAs reveal that no system achieves robust reliability. Furthermore, our diagnostic analysis traces the etiology of these failures to systemic deficits, specifically hallucination propagation and cognitive biases, providing foundational insights to guide future architectural optimization. Data and code are available at https: //github.com/yuhao-zhan/DeepHalluBench.

![image-20260130215808400](assets/main_results.png)
