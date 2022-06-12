"use strict";(self.webpackChunkjackmleitch_com_np=self.webpackChunkjackmleitch_com_np||[]).push([[4202],{3905:function(e,t,n){n.d(t,{Zo:function(){return m},kt:function(){return d}});var r=n(7294);function i(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function a(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){i(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,i=function(e,t){if(null==e)return{};var n,r,i={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(i[n]=e[n]);return i}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(i[n]=e[n])}return i}var l=r.createContext({}),s=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):a(a({},t),e)),n},m=function(e){var t=s(e.components);return r.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},u=r.forwardRef((function(e,t){var n=e.components,i=e.mdxType,o=e.originalType,l=e.parentName,m=c(e,["components","mdxType","originalType","parentName"]),u=s(n),d=i,g=u["".concat(l,".").concat(d)]||u[d]||p[d]||o;return n?r.createElement(g,a(a({ref:t},m),{},{components:n})):r.createElement(g,a({ref:t},m))}));function d(e,t){var n=arguments,i=t&&t.mdxType;if("string"==typeof e||i){var o=n.length,a=new Array(o);a[0]=u;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:i,a[1]=c;for(var s=2;s<o;s++)a[s]=n[s];return r.createElement.apply(null,a)}return r.createElement.apply(null,n)}u.displayName="MDXCreateElement"},6721:function(e,t,n){n.r(t),n.d(t,{assets:function(){return m},contentTitle:function(){return l},default:function(){return d},frontMatter:function(){return c},metadata:function(){return s},toc:function(){return p}});var r=n(7462),i=n(3366),o=(n(7294),n(3905)),a=["components"],c={slug:"Recipe-Recommendation",title:"Building a Recipe Recommendation System",tags:["Python","NLP","Recommendation System"],authors:"jack"},l=void 0,s={permalink:"/PersonalWebsite/blog/Recipe-Recommendation",source:"@site/blog/2021-07-28-RecipeRecomm.md",title:"Building a Recipe Recommendation System",description:"Using Word2Vec, Scikit-Learn, and Streamlit",date:"2021-07-28T00:00:00.000Z",formattedDate:"July 28, 2021",tags:[{label:"Python",permalink:"/PersonalWebsite/blog/tags/python"},{label:"NLP",permalink:"/PersonalWebsite/blog/tags/nlp"},{label:"Recommendation System",permalink:"/PersonalWebsite/blog/tags/recommendation-system"}],readingTime:12.565,truncated:!0,authors:[{name:"Jack Leitch",title:"Machine Learning Engineer",url:"https://github.com/jackmleitch",imageURL:"https://github.com/jackmleitch.png",key:"jack"}],frontMatter:{slug:"Recipe-Recommendation",title:"Building a Recipe Recommendation System",tags:["Python","NLP","Recommendation System"],authors:"jack"},prevItem:{title:"Predicting Strava Kudos",permalink:"/PersonalWebsite/blog/Strava-Kudos"},nextItem:{title:"Automating Mundane Web-Based Tasks With Selenium and Heroku",permalink:"/PersonalWebsite/blog/Strava-AP"}},m={authorsImageUrls:[void 0]},p=[],u={toc:p};function d(e){var t=e.components,c=(0,i.Z)(e,a);return(0,o.kt)("wrapper",(0,r.Z)({},u,c,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"Using Word2Vec, Scikit-Learn, and Streamlit")),(0,o.kt)("p",null,"First things first, If you would like to play around with the finished app. You can here: ",(0,o.kt)("a",{parentName:"p",href:"https://share.streamlit.io/jackmleitch/whatscooking-deployment/streamlit.py"},"https://share.streamlit.io/jackmleitch/whatscooking-deployment/streamlit.py"),"."),(0,o.kt)("p",null,(0,o.kt)("img",{alt:"alt",src:n(5032).Z,width:"1134",height:"801"})),(0,o.kt)("p",null,"In a previous blog post (Building a Recipe Recommendation API using Scikit-Learn, NLTK, Docker, Flask, and Heroku) I wrote about how I went about building a recipe recommendation system. To summarize: I first cleaned and parsed the ingredients for each recipe (for example, 1 diced onion becomes onion), next I ",(0,o.kt)("strong",{parentName:"p"},"encoded each recipe ingredient list using TF-IDF"),". From here I applied a similarity function to find the similarity between ",(0,o.kt)("strong",{parentName:"p"},"ingredients for known recipes and the ingredients given by the end-user"),". Finally, we can get the top-recommended recipes according to the similarity score."))}d.isMDXComponent=!0},5032:function(e,t,n){t.Z=n.p+"assets/images/app-b4b868e72b5712f9ec9ec9e0ee050eb6.png"}}]);