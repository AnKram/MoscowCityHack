function cl(i) {
	console.log(i);
}

APP_URL = 'http://167.172.109.27:4355/';

// дожидаемся полной загрузки страницы
window.onload = function () {
    var input = getById('textarea');
    var button_parse = getById('parse');
    var processed_area = getById('processed');
    var result_area = getById('result');
    
    button_parse.onclick = function() {
		sendApiRequest({'cv': input.value}, 'parse');
    }
}

function insertHtmlById(html_el_id, html) {
	el = getById(html_el_id);
	insertHtmlByEl(el, html);
}

function insertHtmlByEl(el, html) {
	el.innerHTML = html;
}

function getById(html_el_id) {
	return document.getElementById(html_el_id);
}

function getVacanciesHtml(vacancies) {
	html = '<h5>Рекомендации</h5>';
	i = 1;

	for (const [key, cluster] of Object.entries(vacancies)) {
		html += '<h5>Кластер ' + i++ + '</h5>'; 

		for (const [key, vacance] of Object.entries(cluster)) {
			html += '<div class="vacance">';
			html += '<h5>Ключевые навыки</h5>';
			html += '<p>' + vacance.all_skills + '</p>';
			html += '<h5>Рекомендованный курс</h5>';
			html += '<p>' + vacance.course_recommended + '</p>';
			html += '<h5>Вакансия</h5>';
			html += '<p>' + vacance.description + '</p>';
			html += '</div>';
		}
	}
	
	return html;
}

function setResult(json) {
	result = JSON.parse(json);
	
	html = '<h5>Текущий уровень</h5><p>' + result.bucket.student + '</p>';
	el = getById('bucket');
	insertHtmlByEl(el, html);
	
	price = Math.round(result.salary / 1000) * 1000;
	html = '<h5>Рекомендуемая заработная плата</h5><p>' + price + '</p>';
	el = getById('salary');
	insertHtmlByEl(el, html);
	
	html = getVacanciesHtml(result.vacancies);
	el = getById('vacancies');
	insertHtmlByEl(el, html);
	
	
}

function change(el) {
	el.querySelector('.vac_btn').html('')
	el.querySelector('.vac_item').toggleClass('closed');
}

function triggerIvents() {
	processed_area = getById('vacancies');
	elList = processed_area.getElementsByClassName('vacance');
	
	// повесим события на все слова из перевода, по клику отправим в апи
	for (let el of elList) {
		el = getById(el);
		el.addEventListener('click', 
							function(){
								change(this.dataset.word)
							}, 
							false
						);
	}
}

//------API------//
function sendApiRequest(data, method) {
	//$('#loader').show();
	var formData = JSON.stringify(data);
	var xhttp = new XMLHttpRequest();
	xhttp.open("POST", APP_URL + method, false);
	xhttp.setRequestHeader('Content-type', 'application/json;charset=UTF-8');
	xhttp.send(formData);
	
	setResult(xhttp.responseText);
	//$('#loader').hide();
}
