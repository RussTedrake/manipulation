// Loaded after htmlbook/book.js to select Colab for this textbook.
function notebook_link(chapter, notebook, link_text = "") {
  const name = notebook || chapter;
  const path = `book/${chapter}/${name}.ipynb`;
  const url = "https://colab.research.google.com/github/RussTedrake/manipulation/blob/master/"
    + path.split("/").map(encodeURIComponent).join("/");
  const target = `${chapter}_${name}_colab`;
  if (link_text) {
    return `<a href="${url}" target="${target}">${link_text}</a>`;
  }
  return `<p><a href="${url}" target="${target}" style="background:none; border:none;">`
    + '<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></p>';
}
