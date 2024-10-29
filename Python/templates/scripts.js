
function showForm(formId) {
    // Ẩn tất cả các form dự đoán
    document.querySelectorAll('.predict-form').forEach(form => form.classList.add('hidden'));
    // Hiện form dự đoán được chọn
    document.getElementById(formId).classList.remove('hidden');
}

let imgSrcs = []; // Mảng chứa các ảnh cho mỗi loại dữ liệu
let currentImageIndex = 0; // Chỉ số ảnh hiện tại

// Hàm cập nhật hiển thị hình ảnh và nút điều hướng
function updateImage() {
    const imgElement = document.getElementById('imgElement');
    const prevBtn = document.getElementById('prevBtn');
    const nextBtn = document.getElementById('nextBtn');

    // Cập nhật hình ảnh dựa trên chỉ số hiện tại
    if (imgSrcs[currentImageIndex]) {
        imgElement.src = imgSrcs[currentImageIndex];
        imgElement.style.display = 'block';
    } else {
        imgElement.style.display = 'none'; // Ẩn nếu không có hình ảnh
    }

    // Ẩn nút điều hướng nếu chỉ có 1 hình hoặc không có hình
    if (imgSrcs.length <= 1 || !imgSrcs[currentImageIndex]) {
        prevBtn.style.display = 'none';
        nextBtn.style.display = 'none';
    } else {
        prevBtn.style.display = currentImageIndex > 0 ? 'block' : 'none'; // Ẩn nút trái nếu đang ở ảnh đầu tiên
        nextBtn.style.display = currentImageIndex < imgSrcs.length - 1 ? 'block' : 'none'; // Ẩn nút phải nếu đang ở ảnh cuối
    }
}

// Nút mũi tên để chuyển đổi hình ảnh
function prevImage() {
    if (currentImageIndex > 0) {
        currentImageIndex--;
        updateImage();
    }
}

function nextImage() {
    if (currentImageIndex < imgSrcs.length - 1) {
        currentImageIndex++;
        updateImage();
    }
}

function updateDisplayText() {
    const colDisplay = document.getElementById('col_display');
    const imgDiv = document.getElementById('img');
    const selectedType = document.getElementById('select_box_1').value;
    const selectedYear = document.getElementById('select_box_2').value;

    let displayText = "";
    imgSrcs = []; // Khởi tạo lại mảng ảnh cho mỗi lần chọn dữ liệu

    // Văn bản mặc định khi chưa chọn gì
    if (!selectedType && !selectedYear) {
        displayText = "";
        imgSrcs = [];
    } else if (selectedType) {
        switch (selectedType) {
            case 'nguyen_nhan':
                displayText = 
                    "- Từ năm 2022 đến nay, nguyên nhân chính gây tai nạn giao thông ở TPHCM bao gồm " + 
                    "va chạm giữa các phương tiện (chiếm tỷ lệ lớn, đặc biệt là năm 2023), " + 
                    "không tuân thủ quy định giao thông (đặc biệt phổ biến trong năm 2022 nhưng giảm vào 2023), " + 
                    "mất lái, không chú ý quan sát, và say xỉn khi lái xe.<br><br>" +
                    "- Nguyên nhân gây tai nạn giao thông năm 2023 bao gồm: " + 
                    "Va chạm giữa các phương tiện chiếm tỷ lệ cao nhất với 37%, " + 
                    "không tuân thủ quy định giảm nhờ các biện pháp siết chặt giao thông. " + 
                    "Say xỉn chiếm 7%, và các vấn đề như mất lái, không chú ý quan sát cũng tiếp tục tồn tại.<br><br>" +
                    "- Nguyên nhân gây tai nạn giao thông năm 2022 tập trung vào việc " + 
                    "không tuân thủ quy định giao thông chiếm 27% và va chạm giữa các phương tiện chiếm 26%. " + 
                    "Mất lái và không chú ý quan sát cũng là các nguyên nhân phổ biến, lần lượt chiếm 15% và 10%.<br><br>";
                imgSrcs = ["img/diadiem_2023.png", "img/diadiem_2022.png", "img/nguyennhan_2023.png", "img/nguyennhan_2022.png"];

                if (selectedYear === '2023') {
                    displayText = 
                        "**Nguyên nhân tai nạn năm 2023:**<br><br>" +
                        "- Va chạm giữa các phương tiện: Chiếm tỉ lệ cao nhất với 37% tổng vụ án.<br>" +
                        "- Không tuân thủ quy định giao thông: Chiếm khoảng 18% tổng vụ án.<br>" +
                        "- Mất lái: Chiếm khoảng 19% tổng vụ án.<br>" +
                        "- Không chú ý quan sát: Chiếm khoảng 7% tổng vụ án.<br><br>" +
                        "**Trong đó:**<br><br>"+
                        "- Xe máy gây tai nạn chiếm 60,74% tổng vụ tai nạn<br>" +
                        "- Xe ô tô gây tai nạn chiếm 7,05% <br> " +
                        "- Xe tải, xe container và các nguyên nhân khác chiếm 32,21% <br><br>" +
                        "**Địa điểm trọng điểm:**<br><br>" +
                        "- Thủ Đức: Là quận có số vụ tai nạn cao nhất với 12,42% tổng vụ án.<br>" +
                        "- Quốc Lộ 1 và Xa lộ Hà Nội: Các con đường lớn với mật độ giao thông cao, thường xuyên xảy ra tai nạn.<br>" +
                        "- Đường Nguyễn Văn Linh: Nơi có nhiều va chạm do sự lưu thông đông đúc.";
                    imgSrcs = ["img/matdo_2023.png", "img/xemay_2023.png", "img/oto_2023.png", "img/xelon_2023.png"];
                } else if (selectedYear === '2022') {
                    displayText = 
                        "**Nguyên nhân tai nạn năm 2022:**<br><br>" +
                        "- Không tuân thủ quy định giao thông: Chiếm khoảng 27% tổng vụ án.<br>" +
                        "- Va chạm giữa các phương tiện: Chiếm tỉ lệ cao nhất với 26% tổng vụ án.<br>" +
                        "- Mất lái: Chiếm khoảng 15% tổng vụ án.<br>" +
                        "- Không chú ý quan sát: Chiếm khoảng 11% tổng vụ án.<br><br>" +
                        "**Trong đó:**<br><br>"+
                        "- Xe máy gây tai nạn chiếm 85,41% tổng vụ tai nạn<br>" +
                        "- Xe ô tô gây tai nạn chiếm 4,63% <br> " +
                        "- Xe tải, xe container và các nguyên nhân khác chiếm 9,96% <br><br>" +
                        "**Địa điểm trọng điểm:**<br><br>" +
                        "- Thủ Đức: Có số vụ tai nạn cao nhất với 14,24% tổng vụ án.<br>" +
                        "- Quận 7: Chỉ thua Thủ Đức 1 vụ, đặc biệt trên đường Nguyễn Văn Linh.<br>" +
                        "- Đường Phạm Văn Đồng: Cũng là điểm nóng với nhiều vụ va chạm.";
                    imgSrcs = ["img/matdo_2022.png", "img/xemay_2022.png", "img/oto_2022.png", "img/xelon_2022.png"];
                }
                break;

            case 'tuoi':
                displayText = 
                    "- Từ năm 2022 đến nay, độ tuổi gây tai nạn chủ yếu là từ 25 đến 31 tuổi. Đây là độ tuổi tham gia giao thông nhiều"+
                    ", nhưng kinh nghiệm lái xe chưa dày dạn. Tai nạn giảm dần ở độ tuổi từ 32 trở lên.<br><br>"+
                    "- Năm 2023, nhóm tuổi 25 đến 31 vẫn chiếm đa số vụ tai nạn. Tuy nhiên, số vụ tai nạn ở độ tuổi 53 trở lên đã"+
                    " giảm mạnh từ 3,91% xuống 1,34%.<br><br>"+
                    "- Năm 2022, nhóm tuổi 25 đến 31 gây nhiều tai nạn nhất, tiếp theo là nhóm 18 đến 24. Sau đó, tai"+
                    " nạn giảm dần ở các độ tuổi lớn hơn.";
                imgSrcs = ["img/tuoi_tong.jpg"];

                if (selectedYear === '2023') {
                    displayText = 
                        "**Tình hình tai nạn giao thông theo độ tuổi năm 2023:**<br><br>" +
                        "- Nhóm tuổi 18 - 24: Chiếm 7,38% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 25 - 31: Chiếm 37,92% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 32 - 38: Chiếm 29,53% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 39 - 45: Chiếm 19,46% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 46 - 52: Chiếm 4,37% tổng số vụ tai nạn.<br>" + 
                        "- Nhóm tuổi 53 trở lên: Chiếm 1,34% tổng số vụ tai nạn."; 
                    imgSrcs = ["img/tuoi_2023.png"];
                } else if (selectedYear === '2022') {
                    displayText = 
                        "**Tình hình tai nạn giao thông theo độ tuổi năm 2022:**<br><br>" +
                        "- Nhóm tuổi 18 - 24: Chiếm 4,63% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 25 - 31: Chiếm 33,81% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 32 - 38: Chiếm 26,33% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 39 - 45: Chiếm 23,13% tổng số vụ tai nạn.<br>" +
                        "- Nhóm tuổi 46 - 52: Chiếm 8,19% tổng số vụ tai nạn.<br>" + 
                        "- Nhóm tuổi 53 trở lên: Chiếm 3,91% tổng số vụ tai nạn."; 
                    imgSrcs = ["img/tuoi_2022.png"];
                }
                break;

            case 'gio':
                displayText = 
                   "- Tai nạn giao thông thường xảy ra vào buổi chiều và tối, từ 12h đến 24h, đặc biệt là giờ cao điểm."+
                    " Tai nạn ít hơn vào buổi trưa và buổi sáng.<br><br>"+
                    "- Năm 2023, tai nạn vẫn chủ yếu diễn ra vào chiều tối, nhưng buổi trưa có sự tăng nhẹ về số vụ tai nạn."+
                    " Buổi sáng có giảm nhẹ số lượng tai nạn.<br><br>"+
                    "- Năm 2022, tai nạn tập trung vào buổi chiều tối từ 12h đến 24h. Buổi sáng và buổi trưa có số lượng tai nạn ít hơn.";
                imgSrcs = ["img/thang_2023.png", "img/thang_2022.png"];

                if (selectedYear === '2023') {
                    displayText = 
                        "**Tình hình tai nạn giao thông theo giờ năm 2023:**<br><br>" +
                        "- Buổi sáng (5h - 10h): Chiếm 22,52% tổng số vụ tai nạn.<br>" +
                        "- Buổi trưa (10h - 12h): Chiếm 9,04% tổng số vụ tai nạn.<br>" +
                        "- Buổi chiều (12h - 18h): Chiếm 32,44% tổng số vụ tai nạn.<br>" +
                        "- Buổi tối (18h - 24h): Chiếm 35,49% tổng số vụ tai nạn.<br>" +
                        "- Buổi khuya (24h - 5h): Chiếm 0,57% tổng số vụ tai nạn.";
                    imgSrcs = ["img/gio_2023.png"];
                } else if (selectedYear === '2022') {
                    displayText = 
                        "**Tình hình tai nạn giao thông theo giờ năm 2022:**<br><br>" +
                        "- Buổi sáng (5h - 10h): Chiếm 27,44% tổng số vụ tai nạn.<br>" +
                        "- Buổi trưa (10h - 12h): Chiếm 5,14% tổng số vụ tai nạn.<br>" +
                        "- Buổi chiều (12h - 18h): Chiếm 30,05% tổng số vụ tai nạn.<br>" +
                        "- Buổi tối (18h - 24h): Chiếm 37,01% tổng số vụ tai nạn.<br>" +
                        "- Buổi khuya (24h - 5h): Chiếm 0,36% tổng số vụ tai nạn.";
                    imgSrcs = ["img/gio_2022.png"];
                }
                break;
            case 'hau_qua':
                displayText = 
                "- Sau tất cả vụ tai nạn, thiệt hại là thứ đau đớn nhất cho tất cả người liên quan." +
                " Việc cần làm là thống kê số liệu thiệt hại để thức tỉnh ý thức của mọi người khi tham gia giao thông.<br><br>" +
                "- Theo thống kê, năm 2022 có 349 người bị thương và 206 người tử vong do tai nạn giao thông." +
                "- Năm 2023, số người tử vong gia tăng với 251 trường hợp và 489 người bị thương." +
                "- Mỗi vụ tai nạn ảnh hưởng sâu sắc đến gia đình, bạn bè và cộng đồng, gây ra mất mát không thể bù đắp.<br><br>" +
                "- Tai nạn giao thông gây tổn thất lớn về mặt kinh tế, bao gồm chi phí chữa trị, pháp lý, bồi thường, " +
                "cùng với việc sửa chữa hoặc thay thế phương tiện hư hại. Mất đi nguồn lao động, đặc biệt là người trụ cột, " +
                "cũng gây thiệt hại nghiêm trọng.<br><br>" +
                "- Tai nạn thường gây ùn tắc giao thông, ảnh hưởng đến di chuyển của hàng nghìn người và làm giảm năng suất lao động. " +
                "Sự căng thẳng, bực bội từ tình trạng ùn tắc cũng ảnh hưởng tiêu cực đến hoạt động kinh doanh.<br><br>" +
                "- Nạn nhân và gia đình gặp phải vấn đề tâm lý nghiêm trọng như lo âu, sợ hãi, và trầm cảm, " +
                "có thể dẫn đến hội chứng căng thẳng sau chấn thương (PTSD), ảnh hưởng đến sức khỏe tinh thần và chất lượng cuộc sống.";
                imgSrcs = ["img/hauqua.png"];

                break;
            case 'de_xuat':
            displayText = 
                "- Tai nạn giao thông thường xảy ra vào buổi tối muộn, đặc biệt là khi mật độ phương tiện tăng cao."+
                " Cần tăng cường lực lượng cảnh sát và lắp đặt camera giám sát tại các khu vực dễ xảy ra tai nạn.<br><br>"+
                "- Nhóm tuổi từ 25-31 có tỷ lệ tai nạn cao, cần giáo dục an toàn giao thông từ sớm và nâng cao chương trình đào tạo lái xe."+
                " Khuyến khích chương trình học lái xe an toàn cho tài xế trẻ.<br><br>"+
                "- Những nguyên nhân phổ biến gây tai nạn bao gồm vi phạm tốc độ, không chú ý quan sát, và vi phạm quy định giao thông."+
                " Cần mở rộng hệ thống giám sát giao thông thông minh, áp dụng công nghệ giám sát hành vi lái xe và phạt nguội nghiêm khắc.<br><br>"+
                "- Điều kiện đường sá không an toàn cũng là nguyên nhân góp phần gây tai nạn, cần nâng cấp đường bộ, thiết kế lại các giao lộ và cải tiến hệ thống biển báo, đèn tín hiệu.<br><br>"+
                "- Các công nghệ thông minh như hệ thống dự đoán và cảnh báo tai nạn dựa trên machine learning và dữ liệu thời gian thực có thể nâng cao khả năng quản lý và dự đoán tai nạn."+
                " Triển khai ứng dụng di động để cảnh báo nguy cơ tai nạn cho người tham gia giao thông.";
            imgSrcs = ["img/dexuat.png"];

                break;
                
            default:
                // displayText = "<p>Chọn dữ liệu để hiển thị</p>";
                // imgSrc = "";
                break;
        }
    }

    // Cập nhật nội dung hiển thị cho colDisplay
    colDisplay.innerHTML = displayText ? `<p>${displayText}</p>` : 
        "<br>**Lời mở đầu:**<br><br>" +
        "- Xin chào, đây là một ứng dụng nhỏ nhằm mang đến các thông số của cá nhân về đề tài Phân tích tình trạng tai nạn giao thông " +
        "tại TP.HCM: Nguyên nhân, hệ quả và giải pháp.<br><br>" +
        "- Phạm vi thực hiện là tại địa bàn Thành phố Hồ Chí Minh năm 2022, 2023.<br><br>" +
        "- Kết quả và các thông số dữ liệu đều dựa vào bộ dữ liệu thu thập từ cá nhân, mang tính tham khảo cao hơn khẳng định.<br><br>" +
        "- Số liệu được thu thập và xử lý bởi Lê Chí Tài.<br><br>" + 
        "- Chúc bạn một ngày tốt lành và hãy nhớ rằng luôn nâng cao ý thức khi tham gia giao thông nhé."

    imgSrcs = displayText ? imgSrcs : ["img/logo.jpg"];
    
    // Cập nhật nội dung hiển thị cho imgDiv
    currentImageIndex = 0;
    updateImage();
}

// Thêm sự kiện thay đổi cho cả hai select box
document.getElementById('select_box_1').addEventListener('change', updateDisplayText);
document.getElementById('select_box_2').addEventListener('change', updateDisplayText);

// Khởi tạo văn bản hiển thị mặc định
updateDisplayText();