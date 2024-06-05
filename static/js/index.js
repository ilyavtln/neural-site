function update_block(id) {
    let param = document.getElementById(id)

    if (param.classList.contains('show')) {
        param.classList.remove('show')
    }
    else {
        param.classList.add('show')
    }
}