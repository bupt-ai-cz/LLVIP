<!--      表单提交的JS      -->
$(function () {
    $('#rsvp-form').on('submit', function (e) {
        e.preventDefault();
        var data = $(this).serialize();
        console.log('data: ' + data);
        
        var url = 'https://script.google.com/macros/s/AKfycbyKdvyiGN5rrad2hNY48xJvXRC5ox3aA-JhfVmEKRVd0GrFX5EaF-IXKuSsJheI3clPpQ/exec';

        $.post(url, data)
            .done(function (data) {
                console.log(data);
                if (data.result === "error") {
                    show_alert("<strong>Error! </strong> There encounter something wrong, please check your information and resubmit.", "#F66359");
                } else {
                    show_alert("<strong>Success! </strong> We have send the download link to your mailbox, please check it later.", "#2FB986");
                }
            })
            .fail(function (data) {
                console.log(data);
                show_alert("<strong>Error! </strong> There is some issue with the server.", "#F66359");
            });
    });
});


// 提交数据后，显示信息
function show_alert(message, color) {
    document.getElementById('message').innerHTML = message;

    var alert = document.getElementById('alert');
    alert.style.backgroundColor = color;
    alert.style.display = "block";
}
