import React from "react";
import PythonIcon from "@site/static/img/techs/python.svg";
import RIcon from "@site/static/img/techs/R.svg";
import OliveIcon from "@site/static/img/techs/Oliveai.svg";
// import TsIcon from "@site/static/img/techs/typescript.svg";
// import NodeIcon from "@site/static/img/techs/nodejs.svg";

export default [
  {
    date: "July 2022 - Current",
    role: "Machine Learning Engineer - Olive AI",
    location: "Boston, MA",
    website: "https://oliveai.com/",
    icon: <OliveIcon />,
    description: () => (
      <ul>
        <li>
        Designed and implemented architecture, supporting thousands of API calls daily, to automate prior authorizations reducing decision time by an average of 10 days
        </li>
        <li>
        Optimized ML training pipeline by decreasing time to process millions of medical documents from 20 hours to
30 minutes by implementing efficient ETL pipelines in Databricks using PySpark
        </li>
        <li>
        Built a highly scalable medical concept feature generation pipeline using FastAPI and Azure Kubernetes Service
to be used by other core engineering teams
        </li>
        <li>
        Built and maintain internal-use Python packages for data processing, code generation, and feature computation
to add major functionality to AI products
        </li>
      </ul>
    ),
  },
  {
    date: "Feb 2021 - Jun 2021",
    role: "Machine Learning Engineer - IAAD-UK",
    location: "Remote, U.K.",
    website: "https://www.iaad-uk.com",
    icon: <PythonIcon />,
    description: () => (
      <ul>
        <li>
        Built an end-to-end license plate recognition system achieving 94% accuracy 
        </li>
        <li>
        Developed using <b>Python</b>, <b>TensorFlow</b>, <b>OpenCV</b>, and deployed using <b>Docker</b>, <b>FastAPI</b>, and <b>AWS ECS</b>
        </li>
        <li>
        Reduced computing costs from MVP by 46% by cutting image processing time by 67%
        </li>
      </ul>
    ),
  },
  {
    date: "Dec 2017 - Dec 2019",
    role: "Data Science Intern - RSA Insurance Group",
    location: "Horsham, U.K.",
    website: "https://www.rsagroup.com",
    icon: <RIcon />,
    description: () => (
      <ul>
        <li>
        Forecasted motor insurance claims using time series methods (ARIMA) in <b>R</b>
        </li>
        <li>
        Incorporated 3rd party weather data into the model to improve short-term forecasts
        </li>
        <li>
        Integrated R-based forecasts into Excel, for use by actuaries with little coding experience
        </li>
      </ul>
    ),
  },
];
