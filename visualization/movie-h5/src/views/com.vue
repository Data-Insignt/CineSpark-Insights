<template>
    <div class="app-container">
        <!-- <div class="part">
            <chart6 />
        </div>
        <div class="part">
            <chart7 />
        </div>
        <div class="part">
            <chart8 />
        </div> -->
        <div style="font-size:14px; text-align:center; padding:20px;">To Be Continue…..</div>
        <div class="headers">
            <div class="item" @click="jump('/')">Simple Recommendation</div>
            <div class="item" @click="jump('/als')">ALS Model</div>
            <div class="item" @click="jump('/rf')">RF Model</div>
            <div class="item active">Comparing</div>
        </div>
    </div>
</template>
<script>
import chart6 from './charts/chart6'
import chart7 from './charts/chart7'
import chart8 from './charts/chart8'
export default {
    components: {chart6, chart7, chart8},
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