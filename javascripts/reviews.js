// This script contains utilities that allows me 
// to add reviews to the page. Put in simplistic 
// terms, reviews are portions of text, having a tag 'review'.
// An example of such a div is provided below:
// <div class="review" data-author="John Doe">
//     ... review content here ...
// </div>
// The following functions exist in this script:
// - showReview(author): Finds all review divs with the given author and shows them and adds the show-review class to them.
// - hideReview(author): Finds all review divs with the given author and hides them and removes the show-review class from them.
// - hidsAllReviews(): Hides all review divs and removes the show-review class from them.
// - toggleReviewButtonOn(author): Find all buttons with a review-toggle class and data-author attribute equal to the given author 
//  and add the 'active' class to them.
// - toggleAllReviewButtonsOff(author): Find all buttons with a review-toggle class remove the 'active' class from them.


function showReview(author) {
    var reviews = document.querySelectorAll('.review');
    for (var i = 0; i < reviews.length; i++) {
        if (reviews[i].getAttribute('data-author') == author) {
            // The inner HTML is specified in the data-innerhtml attribute

            var inner_html = reviews[i].getAttribute('data-innerhtml');
            reviews[i].innerHTML = inner_html;
            reviews[i].classList.add('show-review');
        }
    }
}

function hideReview(author) {
    var reviews = document.querySelectorAll('.review');
    for (var i = 0; i < reviews.length; i++) {
        if (reviews[i].getAttribute('data-author') == author) {
            reviews[i].classList.remove('show-review');
            reviews[i].innerHTML = '';
        }
    }
}

function hideAllReviews() {
    var reviews = document.querySelectorAll('.review');
    for (var i = 0; i < reviews.length; i++) {
        reviews[i].classList.remove('show-review');
        reviews[i].innerHTML = '';
    }
}


function toggleReviewButtonOn(author) {
    var buttons = document.querySelectorAll('.review-toggle');
    for (var i = 0; i < buttons.length; i++) {
        if (buttons[i].getAttribute('data-author') == author) {
            buttons[i].classList.add('active');
        }
    }
}

function toggleAllReviewButtonsOff() {
    var buttons = document.querySelectorAll('.review-toggle');
    for (var i = 0; i < buttons.length; i++) {
        buttons[i].classList.remove('active');
    }
}

function toggleBtnClicked(author) {
    // This function does the following.
    // 1. Find the first review button with the given author
    // 2. If the button is active:
    //   - Hide all reviews
    //   - Remove the active class from all buttons
    // 3. Else:
    //   - Hide all reviews
    //   - Show the reviews with the given author
    //   - Add the active class to the button with the given author
    var button = document.querySelector('.review-toggle[data-author="' + author + '"]');

    if (button.classList.contains('active')) {
        hideAllReviews();
        toggleAllReviewButtonsOff();
    } else {
        hideAllReviews();
        showReview(author);
        toggleAllReviewButtonsOff();
        toggleReviewButtonOn(author);
    }
}