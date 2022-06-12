"use strict";(self.webpackChunkjackmleitch_com_np=self.webpackChunkjackmleitch_com_np||[]).push([[2765],{3905:function(e,t,n){n.d(t,{Zo:function(){return u},kt:function(){return d}});var r=n(7294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function o(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?o(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):o(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function c(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},o=Object.keys(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);for(r=0;r<o.length;r++)n=o[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var l=r.createContext({}),s=function(e){var t=r.useContext(l),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},u=function(e){var t=s(e.components);return r.createElement(l.Provider,{value:t},e.children)},p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},m=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,o=e.originalType,l=e.parentName,u=c(e,["components","mdxType","originalType","parentName"]),m=s(n),d=a,f=m["".concat(l,".").concat(d)]||m[d]||p[d]||o;return n?r.createElement(f,i(i({ref:t},u),{},{components:n})):r.createElement(f,i({ref:t},u))}));function d(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var o=n.length,i=new Array(o);i[0]=m;var c={};for(var l in t)hasOwnProperty.call(t,l)&&(c[l]=t[l]);c.originalType=e,c.mdxType="string"==typeof e?e:a,i[1]=c;for(var s=2;s<o;s++)i[s]=n[s];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}m.displayName="MDXCreateElement"},182:function(e,t,n){n.r(t),n.d(t,{assets:function(){return u},contentTitle:function(){return l},default:function(){return d},frontMatter:function(){return c},metadata:function(){return s},toc:function(){return p}});var r=n(7462),a=n(3366),o=(n(7294),n(3905)),i=["components"],c={slug:"Strava-Kudos",title:"Predicting Strava Kudos",tags:["Python","Regression","Scikit-Learn","Optuna","XGBoost"],authors:"jack"},l=void 0,s={permalink:"/blog/Strava-Kudos",source:"@site/blog/2021-10-25-StravaKudos.md",title:"Predicting Strava Kudos",description:"An end-to-end data science project, from data collection to model deployment, aimed at predicting user interaction on Strava activities based on the given activity\u2019s attributes.",date:"2021-10-25T00:00:00.000Z",formattedDate:"October 25, 2021",tags:[{label:"Python",permalink:"/blog/tags/python"},{label:"Regression",permalink:"/blog/tags/regression"},{label:"Scikit-Learn",permalink:"/blog/tags/scikit-learn"},{label:"Optuna",permalink:"/blog/tags/optuna"},{label:"XGBoost",permalink:"/blog/tags/xg-boost"}],readingTime:20.325,truncated:!0,authors:[{name:"Jack Leitch",title:"Machine Learning Engineer",url:"https://github.com/jackmleitch",imageURL:"https://github.com/jackmleitch.png",key:"jack"}],frontMatter:{slug:"Strava-Kudos",title:"Predicting Strava Kudos",tags:["Python","Regression","Scikit-Learn","Optuna","XGBoost"],authors:"jack"},prevItem:{title:"Building Interpretable Models on Imbalanced Data",permalink:"/blog/Churn-Prediction"},nextItem:{title:"Building a Recipe Recommendation System",permalink:"/blog/Recipe-Recommendation"}},u={authorsImageUrls:[void 0]},p=[],m={toc:p};function d(e){var t=e.components,n=(0,a.Z)(e,i);return(0,o.kt)("wrapper",(0,r.Z)({},m,n,{components:t,mdxType:"MDXLayout"}),(0,o.kt)("p",null,(0,o.kt)("strong",{parentName:"p"},"An end-to-end data science project, from data collection to model deployment, aimed at predicting user interaction on Strava activities based on the given activity\u2019s attributes.")),(0,o.kt)("p",null,"Strava is a service for tracking human exercise which incorporates social network type features. It is mostly used for cycling and running, with an emphasis on using GPS data. A typical Strava post from myself is shown below and we can see that it contains a lot of information: distance, moving time, pace, elevation, weather, GPS route, who I ran with, etc., etc."))}d.isMDXComponent=!0}}]);