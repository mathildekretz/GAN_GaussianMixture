# Gaussian Mixture GAN - DataLabAssignement2

## IASD Master Program 2023/2024 - PSL Research University

### About this project

This project is the second homework assignment for the Data Science Lab class of the IASD (Artificial Intelligence, Systems and Data) Master Program 2023/2024, at PSL Research University (Université PSL).

*The project achieved the following objectives:*
- Improved the digit generation of the MNIST dataset by employing Gaussian Mixture Generative Adversarial Networks (GM-GAN) in unsupervised learning scenarios, expanding on a well-optimized vanilla GAN.
- Established a strong baseline with the vanilla GAN through meticulous hyperparameter adjustments, providing valuable insights for refining a static GM-GAN.
- Generated a more diverse set of aesthetically pleasing digits at a higher frequency using the final Gaussian Mixture model and recommended hyperparameters identified through the tuning analysis.

## General information

The report can be viewed in the [datalab_report_GAN.pdf](datalab_report_GAN.pdf) file. 

The rest of the instructions can be found below. If you want to copy and recreate this project, or test it for yourself, some important information to know.

**generate.py**
Use the file *generate.py* to generate 10000 samples of MNIST in the folder samples. 
Example:
  > python3 generate.py --bacth_size 64

---

### Acknowledgement

This project was made possible with the guidance and support of the following :

- **Prof. Benjamin Negrevergne**
  - Professor at *Université Paris-Dauphine, PSL*
  - Researcher in the *MILES Team* at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
  - Co-director of the IASD Master Program with Olivier Cappé

- **Alexandre Vérine**
  - PhD candidate at *LAMSADE, UMR 7243* at *Université Paris-Dauphine, PSL* and *Université PSL*
 
This project was a group project, and was made possible thanks to the collaboration of :

- **Mathilde Kretzt**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Amélie Leroy**, *IASD Master Program 2023/2024 student, at PSL Research University*
- **Marius Roger**, *IASD Master Program 2023/2024 student, at PSL Research University*
