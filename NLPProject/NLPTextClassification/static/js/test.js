
var ZXXFILE = {
    fileInput: null,                //html file控件
    dragDrop: null,                 //拖拽敏感区域
    upButton: null,                 //提交按钮
    url: "",                        //ajax地址
    fileFilter: [],                 //过滤后的文件数组
    filter: function(files) {       //选择文件组的过滤方法
        return files;   
    },
    onSelect: function() {},        //文件选择后
    onDelete: function() {},        //文件删除后
    onDragOver: function() {},      //文件拖拽到敏感区域时
    onDragLeave: function() {}, //文件离开到敏感区域时
    onProgress: function() {},      //文件上传进度
    onSuccess: function() {},       //文件上传成功时
    onFailure: function() {},       //文件上传失败时,
    onComplete: function() {},      //文件全部上传完毕时
    
    /* 开发参数和内置方法分界线 */
    
    //文件拖放
    funDragHover: function(e) {
        e.stopPropagation();
        e.preventDefault();
        this[e.type === "dragover"? "onDragOver": "onDragLeave"].call(e.target);
        return this;
    },
    //获取选择文件，file控件或拖放
    funGetFiles: function(e) {
        // 取消鼠标经过样式
        this.funDragHover(e);
                
        // 获取文件列表对象
        var files = e.target.files || e.dataTransfer.files;
        //继续添加文件
        this.fileFilter = this.fileFilter.concat(this.filter(files));
        this.funDealFiles();
        return this;
    },
    
    //选中文件的处理与回调
    funDealFiles: function() {
        for (var i = 0, file; file = this.fileFilter[i]; i++) {
            //增加唯一索引值
            file.index = i;
        }
        //执行选择回调
        this.onSelect(this.fileFilter);
        return this;
    },
    
    //删除对应的文件
    funDeleteFile: function(fileDelete) {
        var arrFile = [];
        for (var i = 0, file; file = this.fileFilter[i]; i++) {
            if (file != fileDelete) {
                arrFile.push(file);
            } else {
                this.onDelete(fileDelete);  
            }
        }
        this.fileFilter = arrFile;
        return this;
    },
    
    //文件上传
    funUploadFile: function() {
		alert('!!!');	
        var self = this;    
        if (location.host.indexOf("sitepointstatic") >= 0) {
            //非站点服务器上运行
            return; 
        }
        for (var i = 0, file; file = this.fileFilter[i]; i++) {
            (function(file) {
                
                //使用ajax异步上传，暂时不考虑兼容性
                //必须使用post才能提交文件类型的数据，即大量的数据
                // xmlHttp.open("post","xxx.do");
                //发送表单数据，然后服务端使用myfile这个名称接收即可
                // xml.send(formData);
                var xhr = new XMLHttpRequest();
                if (xhr.upload) {
                    // 上传中
                    xhr.upload.addEventListener("progress", function(e) {
                        self.onProgress(file, e.loaded, e.total);
                    }, false);
                    var formData=new FormData();
                    //相当于 <input type=file name='myfile' />
                    formData.append("file",file);
                    // 文件上传成功或是失败
                    xhr.onreadystatechange = function(e) {
                        if (xhr.readyState == 4) {
                            if (xhr.status == 200) {
                                self.onSuccess(file, xhr.responseText);
                                self.onComplete();
                                // self.funDeleteFile(file);
                                if (!self.fileFilter.length) {
                                    //全部完毕
                                    self.onComplete();  
                                }
                            } else {
                                self.onFailure(file, xhr.responseText);     
                            }
                        }
                    };
        
                    // 开始上传
                    xhr.open("POST", self.url, true);
                    // xhr.setRequestHeader("X_FILENAME", encodeURIComponent(file.name));
                    xhr.send(formData);
                }   
            })(file);   
        }   
            
    },
    
    init: function() {
        var self = this;
        
        if (this.dragDrop) {
            this.dragDrop.addEventListener("dragover", function(e) { self.funDragHover(e); }, false);
            this.dragDrop.addEventListener("dragleave", function(e) { self.funDragHover(e); }, false);
            this.dragDrop.addEventListener("drop", function(e) { self.funGetFiles(e); }, false);
        }
        
        //文件选择控件选择
        if (this.fileInput) {
            this.fileInput.addEventListener("change", function(e) { self.funGetFiles(e); }, false); 
        }
        
        //上传按钮提交
        if (this.upButton) {
            this.upButton.addEventListener("click", function(e) { self.funUploadFile(e); }, false); 
        } 
    }
};
var params = {
	fileInput: $("#fileText").get(0),
	dragDrop: $("#textInputArea").get(0),
	upButton: $("#fileSubmit").get(0),
	url: $("#uploadForm").attr("action"),

	//filter: function(files) {
	//	var arrFiles = [];
	//	for (var i = 0, file; file = files[i]; i++) {
	//		if (file.type.indexOf("image") == 0) {
	//			arrFiles.push(file);	
	//		} else {
	//			alert('文件"' + file.name + '"不是图片。');	
	//		}
	//	}
	//	return arrFiles;
	//},
	onSelect: function(files) {
		var html = '', i = 0;
		$("#preview").html('<div class="upload_loading"></div>');
		var funAppendImage = function() {
			file = files[i];
			if (file) {
				var reader = new FileReader()
				reader.onload = function(e) {
					html = html + '<div id="uploadList_'+ i +'" class="upload_append_list"><p><strong>' + file.name + '</strong>'+ 
						'<a href="javascript:" class="upload_delete" title="删除" data-index="'+ i +'">删除</a><br />' +
						'<img name = "docfile" id="uploadImage_' + i + '" src="' + e.target.result + '" class="upload_image" /></p>'+
						'<span id="uploadProgress_' + i + '" class="upload_progress">'+'<p>Analyse</p><span></span>'+'<span></span>'+'<span></span>'+'</span>' +
					'</div>'+'<div id="result" class="upload_result"></div>';
					
					i++;
					funAppendImage();
				}
				reader.readAsDataURL(file);
			} else {
				$("#preview").html(html);
				if (html) {
					//删除方法
					$(".upload_delete").click(function() {
						ZXXFILE.funDeleteFile(files[parseInt($(this).attr("data-index"))]);
						return false;	
					});
					//提交按钮显示
					$("#fileSubmit").show();	
				} else {
					//提交按钮隐藏
					$("#fileSubmit").hide();	
				}
			}
		};
		funAppendImage();		
	},
	
	onDelete: function(file) {
		$("#uploadList_" + file.index).fadeOut();
		$("#fileSubmit").hide();
		$("#result").hide();
	},
	onDragOver: function() {
		$(this).addClass("upload_drag_hover");
	},
	onDragLeave: function() {
		$(this).removeClass("upload_drag_hover");
	},
	onProgress: function(file, loaded, total) {
		eleProgress = $("#uploadProgress_" + file.index);
		// eleProgress.show().html("Analyse");
        eleProgress.show();
		// $("#uploadProgress_" + file.index).append("<span>Analyse</span>")
	},
	onSuccess: function(file, response) {
		$("#result").append(response);
	},
	onFailure: function(file) {
		$("#uploadInf").append("<p>图片" + file.name + "上传失败！</p>");	
		$("#uploadImage_" + file.index).css("opacity", 0.2);
	},
	onComplete: function() {
		//提交按钮隐藏
		$("#fileSubmit").hide();
		eleProgress.hide();
		//file控件value置空
		$("#fileText").val("");
		// $("#uploadInf").append("<p>当前图片全部上传完毕，可继续添加上传。</p>");
	}
};
ZXXFILE = $.extend(ZXXFILE, params);
ZXXFILE.init();