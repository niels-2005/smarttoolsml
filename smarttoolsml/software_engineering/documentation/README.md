# Project Documentation

## Overview

This project aims to [briefly describe the purpose of the project].  
The main goals are [explain key objectives], and it is intended for [describe target users].  
Potential use cases include [list possible applications].  

---

## Key Questions to Address

Try to answer the following questions in your documentation:

- Why did you select the data you used in this project?  
- What are the assumptions you are making about your data?  
- Why did you choose this analysis method rather than another?  
- Are there circumstances where this analysis method does not work?  
- What (if any) shortcuts did you take that could be improved later?  
- What are some other avenues for future experimentation that you would suggest to anyone who works on this project in the future?  
- What are the lessons you learned from this project?  

---

## Documentation Within the Codebase

- Names of functions, classes, and modules give information on what you should expect that piece of code to do.  
- Comments make a small individual point that adds extra information, similar to a footnote in a book.  
- Docstrings give a longer overview of what a function or class does, including details of any edge cases.  
- API documentation shows what each API endpoint expects as its input and returns for its output.  
- Longer documents such as README’s and tutorials give an overview of how to use all the code in a project.  

---

## README’s, Tutorials, and Other Longer Documents

Consider including the following in your project’s documentation:

- **A short overview** of your project, which could be a single paragraph.  
  - Think of this as an "executive summary".  
  - Include the overall goals of the project, who should use it, and the use cases.  
- **How to get started** using the project and how to navigate it.  
  - This could take the form of a notebook tutorial.  
- **Project caveats or limitations** that users should be aware of.  
  - For example, if your code only works with data from 2023 or earlier, highlight this in the introduction rather than requiring users to dig into the code comments.  
- **Next steps** for the project.  
  - Even if you have finished working on it, note good next steps for anyone who picks up this work in the future.  

---

## Documenting Machine Learning Experiments

In machine learning projects, you'll try out many different models, datasets, and hyperparameters in search of the model that makes the best predictions according to the evaluation metrics you choose.  

Since the number of hyperparameters can grow very large, it's important to document what combinations you have tried. This will help if you come back to your work in the future, and it will also help anyone else who works on your project.  

To do this, ensure you track all the variables that change in each iteration of your experiment. This allows for experiment reproducibility and helps others understand what has been tested. You should also record any assumptions you are making along the way.  

### Key aspects to document:

- The data you used to train the model  
- The training/validation/test split  
- The feature engineering choices you made  
- The model hyperparameters (such as the regularization in a logistic regression model or learning rate for a neural network)  
- The metrics you are evaluating your model on, such as accuracy, precision, and recall  

---

## Tools for Tracking ML Experiments

- **Weights and Biases**  
- **MLflow**  
