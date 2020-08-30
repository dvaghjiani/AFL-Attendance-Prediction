It's no secret that Australians love their sport. If you go to any social gathering here, it's highly likely that you'll catch at least a snippet of a conversation about sport, whether it be cricket, tennis, soccer or some other sport. However nothing quite matches Australians' passion for Australian Rules Football. The AFL (Australian Football League) is the sport's highest-level league and regularly attracts match crowds in excess of 60,000 people.

Whilst watching a blockbuster match one day, I noticed that the attendance was significantly less than expected for such a highly-anticipated game. This got me thinking about all the possible factors that influence crowd sizes, from the ladder positions of the teams to the match-day weather conditions. Being the kind of guy who loves data-backed answers to questions, I decided to make a machine learning model for it using the Python programming language.

An AFL match attendance predictor could be valuable for companies and key decision makers in a variety of industries. Some use cases include:

- Forecasting stadium revenue
- Valuing stadium advertising opportunities
- Logistics and staff planning at stadiums
- Forecasting and planning for independent businesses (e.g. restaurants and cafés) who operate near stadiums

In fact, I came across an internet forum post from somebody who was operating a café near Marvel Stadium in Melbourne and was searching for a tool to estimate the amount of foot-traffic his café would receive on future AFL match days (a suitable proxy for this would be the attendance at Marvel Stadium games). Here's what he wrote: _"Could anyone point me in the right direction of where I would find an estimated crowd attendance figure in advance before the game, particularly for [Marvel] Stadium? My reason being, I currently manage a cafe near the stadium and would like to get an idea of the possible foot traffic passing my cafe on game day."_

In this blog post, I present a non-technical discussion of my analysis and the models that I created, along with a summary of the results.

> _For readers who aren't familiar with the typical machine learning workflow, I'll use an analogy of a school teacher teaching a student a new concept to help you understand the core idea behind each step of the process._


![Image](/MCG crowd.jpg)
*image_caption*





## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/dvaghjiani/Loan-Default-Prediction-Project/edit/master/docs/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/dvaghjiani/Loan-Default-Prediction-Project/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
