import React from "react";
import PublicImage from "./PublicImage";
import styled from "styled-components";

const skills = [
  "python",
  "docker",
  "azure",
  "aws",
  "postgresql",
  "github",
  "R",
  "linux",
  
];
const Container = styled.div`
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
`;
const Item = styled.div`
  width: 5rem;
  height: 5rem;
  padding: 0.5rem;
  transition: width 0.5s;

  &:hover {
    width: 4rem;
    height: 4rem;
  }
`;
const Image = styled(PublicImage)``;

const Skills = () => {
  return (
    <Container>
      {skills.map((skill, index) => {
        return (
          <Item key={index}>
            <a
              href={`https://github.com/jackmleitch?tab=repositories&q=${skill}&type=&language=`}
              target="_blank"
            >
              <Image name={skill} alt={skill} />
            </a>
          </Item>
        );
      })}
    </Container>
  );
};
export default Skills;