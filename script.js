<!--      表单提交的JS      -->
$(function () {
    $('#rsvp-form').on('submit', function (e) {
        e.preventDefault();

        if (google_validation() === false) {
            show_alert("<strong>Warning! </strong> Please do a test.", "#F66359");
        } else {
            var data = $(this).serialize();
            // console.log('data: ' + data);
            var url = 'https://script.google.com/macros/s/AKfycbyKdvyiGN5rrad2hNY48xJvXRC5ox3aA-JhfVmEKRVd0GrFX5EaF-IXKuSsJheI3clPpQ/exec';

            show_alert("<strong>Hold on! </strong> We are storing your information.", "#47A8F5");

            $.post(url, data)
                .done(function (data) {
                    // console.log(data);
                    if (data.result === "error") {
                        show_alert("<strong>Error! </strong> There encounters something wrong, please check your information and resubmit.", "#F66359");
                    } else {
                        show_alert("<strong>Success! </strong> We have sent the download link to your mailbox, please check it later.", "#2FB986");
                    }
                })
                .fail(function (data) {
                    // console.log(data);
                    show_alert("<strong>Error! </strong> There is some issue with the server.", "#F66359");
                });
        }
    });
});


// 提交数据后，显示信息
function show_alert(message, color) {
    document.getElementById('message').innerHTML = message;

    var alert = document.getElementById('alert');
    alert.style.backgroundColor = color;
    alert.style.display = "block";

    // var alert_list = document.getElementById('alert_list');
    //
    // var alert_div = document.createElement('div');
    // alert_div.className = 'alert';
    // alert_div.style.display = 'block';
    // alert_div.style.backgroundColor = color;
    //
    // var span_btn = document.createElement('span');
    // span_btn.className = 'closebtn';
    // span_btn.onclick = function () {
    //     this.parentElement.style.display = 'none';
    // };
    // span_btn.innerText = "&times";
    //
    // var span_message = document.createElement('span');
    // span_message.innerHTML = message;
    //
    // alert_div.appendChild(span_btn);
    // alert_div.appendChild(span_message);
    //
    // alert_list.appendChild(alert_div);
}

// google validation
function onloadCallback() {
    widgetId = grecaptcha.render('validation', {
        "sitekey": "6LcU6F8bAAAAACGktwCtXY7e1XfIBVjjnqtwjPxi",
    });
}

function google_validation() {
    var response = grecaptcha.getResponse(widgetId);
    console.log('response: ' + response);

    return response.length !== 0;
}
