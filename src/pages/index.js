import React from "react";
import classnames from "classnames";
import Layout from "@theme/Layout";
import useDocusaurusContext from "@docusaurus/useDocusaurusContext";
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./styles.module.css";
import Skills from "../../components/Skills";
import styled from "styled-components";
const features = [
  {
    title: <>About Me</>,
    imageUrl: "img/favicon.ico",
    description: (
      <p>
        Hi, it's me, <b> Jack Leitch</b>. I am a Machine Learning Engineer 
        with experience in statistical learning, computer vision, and NLP.
        <br />
        <br />
        I am ambitious to continuously develop in ML and Data Science, and apply
        this knowledge to a vast range of fields.
      </p>
    ),
  },
  {
    title: <>Experience</>,
    imageUrl: "img/undraw_docusaurus_tree.svg",
    description: (
      <p>
        I have worked at a tech startup where I built an end-to-end license 
        plate recognition system using Tensorflow. I also build data science 
        projects in my spare time on topics that interest me, for example, my Strava 
        Kudos Predictor combines my love for data and running. You can find my projects on 
        my <a href="https://github.com/jackmleitch">Github page</a>.
      </p>
    ),
  },
  {
    title: <>Skills</>,
    imageUrl: "img/undraw_docusaurus_react.svg",
    description: <Skills />,
  },
];

const HeroHeader = styled("header")`
  background: url("img/edinburgh_skyline.png") no-repeat center;
  background-size: cover;
`;
function Feature({ imageUrl, title, description }) {
  const imgUrl = useBaseUrl(imageUrl);
  return (
    <div className={classnames("col col--4", styles.feature)}>
      {imgUrl && (
        <div className="text--center">
          <img className={styles.featureImage} src={imgUrl} alt={title} />
        </div>
      )}
      <h3>{title}</h3>
      {description}
    </div>
  );
}

function Home() {
  const context = useDocusaurusContext();
  const { siteConfig = {} } = context;
  return (
    <Layout
      title={`Hello from ${siteConfig.title}`}
      description="Machine learning engineer from the U.K."
    >
      <HeroHeader
        className={classnames("hero hero--primary", styles.heroBanner)}
      >
        <div className="container image-bg">
          <h1 className="hero__title">{siteConfig.title}</h1>
          <p className="hero__subtitle">{siteConfig.tagline}</p>
          <div className={styles.buttons}>
            <a
              className={classnames(
                "button   button--primary button--lg",
                styles.getStarted
              )}
              href={useBaseUrl("jackleitch_resume.pdf")}
            >
              Download Resume
            </a>
          </div>
        </div>
      </HeroHeader>
      <main>
        {features && features.length > 0 && (
          <section className={styles.features}>
            <div className="container">
              <div className="row">
                {features.map((props, idx) => (
                  <Feature key={idx} {...props} />
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </Layout>
  );
}

export default Home;

