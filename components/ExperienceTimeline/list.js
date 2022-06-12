import React from "react";
import PythonIcon from "@site/static/img/techs/python.svg";
import RIcon from "@site/static/img/techs/R.svg";
// import TsIcon from "@site/static/img/techs/typescript.svg";
// import NodeIcon from "@site/static/img/techs/nodejs.svg";

export default [
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
        Developed using <b>Python</b>, <b>TensorFlow</b>, <b>OpenCV</b>, and deployed using <b>AWS Lambda</b>
        </li>
        <li>
        Reduced computing costs from MVP by 76% by cutting image processing time by 87%
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
