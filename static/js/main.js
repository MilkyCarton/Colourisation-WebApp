function formBox(colour, message){
    if (colour == "success" && message == null ){
        $(".formError").delay(500).queue(function() {
            $(this).html("Success!").dequeue().attr("class", "formError success").animate({height: "45px", marginBottom: "20px"}, 500 ).delay(1000).animate({height: "0px", marginBottom: "0px"}, 500 ).queue(function() {
                $(this).attr("class", "formError").html("").dequeue();
            });
        })
    } else {
        $(".formError").html(message).attr("class", "formError " + colour).animate({height: "45px", marginBottom: "20px"}, 500 );
    }
}

function start(){
    $("#upload").val(null);
    $("#url").val(null);
    $("#uploadPage").show();
    $("#uploading").hide();
    $("#result").hide();
}

function uploading(){
    $("#uploadPage").hide();
    $("#uploading").show();
    $("#result").hide();
}

function success(){
    $("#uploadPage").hide();
    $("#uploading").hide();
    $("#result").show();
}

$('#upload').on('change', function () {

    var formData = new FormData();
    var files = $("#upload")[0].files;

    if (files.length == 0){
        formBox("error", "You must select a file or enter an image url.");
        return;
    }

    formData.append('request', "upload");
    formData.append('file', files[0]);

    var error = false;

    uploading();

    $.ajax({
        url: "/upload-image",
        contentType: false,
        processData: false,
        method: 'POST',
        data: formData
    }).done(function(data) {
        if(data.response !== null && data.response === true){
            success();
            $("#resultImage").attr("src", "data:image/jpeg;base64,"+data.after);
            $("#download").attr("href", data.url).attr("download", data.filename);
        } else {
            if(data.error){
                start();
                formBox("error", data.error);
            }
        }
      });

});

$('#urlSubmit').on('click', function () {

    var formData = new FormData();
    var url = $("#url").val();

    if (url.length == 0){
        formBox("error", "You must enter an image url.");
        return;
    }

    formData.append('request', 'url');
    formData.append('url', url);

    uploading();

    var error = false;

    $.ajax({
        url: "/upload-image",
        contentType: false,
        processData: false,
        method: 'POST',
        data: formData
    }).done(function(data) {
        if(data.response !== null && data.response === true){
            success();
            $("#resultImage").attr("src", "data:image/jpeg;base64,"+data.after);
            $("#download").attr("href", data.url).attr("download", data.filename);
        } else {
            if(data.error){
                start();
                formBox("error", data.error);
            }
        }
      });

});