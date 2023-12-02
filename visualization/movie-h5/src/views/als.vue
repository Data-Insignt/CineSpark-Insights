<template>
    <div class="app-container">
        <div class="part">
            <chart-three />
        </div>
        <div class="part">
            <chart-four />
        </div>
        <div class="part">
            <chart-five />
        </div>
        <div class="headers">
            <div class="item" @click="jump('/')">Simple Recommendation</div>
            <div class="item active">ALS Model</div>
            <div class="item" @click="jump('/rf')">RF Model</div>
            <div class="item" @click="jump('/com')">Comparing</div>
        </div>
    </div>
</template>
<script>
import chartThree from './charts/chartThree'
import chartFour from './charts/chartFour'
import chartFive from './charts/chartFive'
export default {
    components: {chartThree, chartFour, chartFive},
    methods: {
        jump(url) {
            this.$router.push(url)
        }
    }
}
</script>
<style scoped>
.app-container {
    padding-bottom: 50px;
}
.part {
    background: #fff;
    padding: 10px 0;
    width: 95%;
    margin: 10px auto;
    border-radius: 5px;
}
.headers {
    display: flex;
    align-items: center;
    background: #ffff;
    color: #333;
    font-size: 14px;
    text-align: center;
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    height: 50px;
    border-top: 1px solid #eee;
}
.headers .item {
    flex: 1;
    padding: 5px;
}
.headers .item.active {
    color: #e63737;
}
</style>