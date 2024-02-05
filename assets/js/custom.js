(function ($) {
    "use strict";
    $(document).ready(function () {

        /* ¸ü¶àÏÂÔØ£ºHttp://www.bootstrapmb.com
       Jquery Mobile Menu
       ============================*/
        $('#main-menu').meanmenu({
            meanMenuContainer: '.mobile-nav-menu',
            meanScreenWidth: "991",
            meanExpand: ['<i class="fa-solid fa-angle-right"></i>'],
        });

        /*
       Jquery Sidebar Toggle
       ============================*/
        $(".mobile-menu-toggle-btn").on("click", function () {
            $(".menu-sidebar-area").addClass("active");
            $(".body-overlay").addClass("active");
        });
        $(".menu-sidebar-close-btn").on("click", function () {
            $(".menu-sidebar-area").removeClass("active");
            $(".body-overlay").removeClass("active");
        });

        /*
       Jquery Body Overlay
       ============================*/
        $(".body-overlay").on("click", function () {
            $(".menu-sidebar-area").removeClass("active");
            $(".body-overlay").removeClass("active");
        });
        /*
        Stikey Js
        ============================*/

        const body = document.body;
        const html = document.documentElement;
        const height = Math.max(body.scrollHeight, body.offsetHeight, html.clientHeight, html.scrollHeight, html.offsetHeight);

        if (100 < $(window).scrollTop()) {
            $(".header-menu-area.sticky-header").addClass("sticky_menu");
        }

        if( height  > 1400 ) {
            const nav = $(".header-menu-area.sticky-header");
            let scrolled = false;
            $(window).scroll(function () {
                if (100 < $(window).scrollTop() && !scrolled) {
                    nav
                        .addClass("sticky_menu animated fadeIn")
                        .animate({"margin-top": "0px"});
                    scrolled = true;
                }
                if (100 > $(window).scrollTop() && scrolled) {
                    nav.removeClass("sticky_menu animated fadeIn").css("margin-top", "0px");
                    scrolled = false;
                }
            });
        }

        /*
        Jquery Empty Post Content Hide
        ============================*/
        $('.blog-area .post-content p').filter(function() {
            return /\u00A0/.test($(this).text());
        }).hide();

        /*
        Typing animation 
        ============================*/
        
    var TxtType = function(el, toRotate, period) {
        this.toRotate = toRotate;
        this.el = el;
        this.loopNum = 0;
        this.period = parseInt(period, 10) || 2000;
        this.txt = '';
        this.tick();
        this.isDeleting = false;
    };

    TxtType.prototype.tick = function() {
        var i = this.loopNum % this.toRotate.length;
        var fullTxt = this.toRotate[i];

        if (this.isDeleting) {
        this.txt = fullTxt.substring(0, this.txt.length - 1);
        } else {
        this.txt = fullTxt.substring(0, this.txt.length + 1);
        }

        this.el.innerHTML = '<span class="wrap">'+this.txt+'</span>';

        var that = this;
        var delta = 200 - Math.random() * 100;

        if (this.isDeleting) { delta /= 2; }

        if (!this.isDeleting && this.txt === fullTxt) {
        delta = this.period;
        this.isDeleting = true;
        } else if (this.isDeleting && this.txt === '') {
        this.isDeleting = false;
        this.loopNum++;
        delta = 500;
        }

        setTimeout(function() {
        that.tick();
        }, delta);
    };

    window.onload = function() {
        var elements = document.getElementsByClassName('typewrite');
        for (var i=0; i<elements.length; i++) {
            var toRotate = elements[i].getAttribute('data-type');
            var period = elements[i].getAttribute('data-period');
            if (toRotate) {
              new TxtType(elements[i], JSON.parse(toRotate), period);
            }
        }
        // INJECT CSS
        var css = document.createElement("style");
        css.type = "text/css";
        css.innerHTML = ".typewrite > .wrap { border-right: 0.08em solid #FFFFFF}";
        document.body.appendChild(css);
    };


        /*
        Skill Progress Bar Js
        ============================*/
        $('.skill-progressbar').one('inview', function(event, isInView) {
            if (isInView) {
                $('.progress-inner').each(function() {
                    $(this).find('.progress-content').animate({
                        width:$(this).attr('data-percentage')
                    },2000);

                    $(this).find('.progress-number-count').animate(
                        {left:$(this).attr('data-percentage')},
                        {
                            duration: 2000,
                            step: function(now) {
                                let data = Math.round(now);
                                $(this).find('.progress-percent').html(data + '%');
                            }
                        });
                });

            }
        });

        /*
       Slider
       ============================*/
        $("#slider-wrapper").slick({
            slidesToShow: 1,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1500,
            rtl: false,
            vertical: true,
            prevArrow:
                "<button type='button' class='slider-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='slider-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 767,
                    settings: {
                        autoplay: true,
                        vertical: false,
                    },
                },
            ],
        });

        $('#trigger_header_slider_prev').on('click', function() {
            console.log("Slider clicked");
            $('#slider-wrapper .prev-btn').trigger('click');
        });
        $('#trigger_header_slider_next').on('click', function() {
            console.log("Slider clicked");
            $('#slider-wrapper .next-btn').trigger('click');
        });

        /*
       Info Card Slider
       ============================*/
        $("#info_card_slider_wrapper").slick({
            slidesToShow: 3,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='info-card-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='info-card-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 992,
                    settings: {
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 768,
                    settings: {
                        slidesToShow: 1,
                    },
                },
            ],
        });

        /*
       Team Slider
       ============================*/
        $("#team_slider_wrapper").slick({
            slidesToShow: 3,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: false,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            responsive: [
                {
                    breakpoint: 992,
                    settings: {
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 768,
                    settings: {
                        slidesToShow: 1,
                    },
                },
            ],
        });

         /*
       Brand Slider
       ============================*/
        $(".brand-slider-wrapper").slick({
            slidesToShow: 5,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: false,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1500,
            rtl: false,
            responsive: [
                {
                    breakpoint: 992,
                    settings: {
                        slidesToShow: 4,
                    },
                },
                {
                    breakpoint: 768,
                    settings: {
                        slidesToShow: 2,
                    },
                },
            ],
        });

        /*
       Testimonial Slider
       ============================*/
        $("#home3_testimonial").slick({
            slidesToShow: 2,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='testimonial-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='testimonial-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 1023,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                    },
                }
            ],
        });


        /*
       Testimonial Slider
       ============================*/
        $("#testimonial_two").slick({
            slidesToShow: 2,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='testimonial-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='testimonial-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 1200,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                    },
                }
            ],
        });

        $('#trigger_testimonial_prev').on('click', function() {
            $('.testimonial-slider-wrapper .prev-btn').trigger('click');
        });
        $('#trigger_testimonial_next').on('click', function() {
            $('.testimonial-slider-wrapper .next-btn').trigger('click');
        });

        $("#testimonial_three").slick({
            slidesToShow: 1,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='testimonial-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='testimonial-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 767,
                    settings: {
                        autoplay: true,
                    },
                },
            ],
        });

        /*
       Location Slider
       ============================*/
        $("#location-slider-wrapper").slick({
            slidesToShow: 4,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='location-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='location-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 1200,
                    settings: {
                        autoplay: true,
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 767,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                    },
                }
            ],
        });

        $('#trigger_location_prev').on('click', function() {
            $('.location-slider-wrapper .prev-btn').trigger('click');
        });
        $('#trigger_location_next').on('click', function() {
            $('.location-slider-wrapper .next-btn').trigger('click');
        });

        //Location slider-2
        $("#location-slider-wrapper-2").slick({
            slidesToShow: 3,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: true,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='location-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='location-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 1200,
                    settings: {
                        autoplay: true,
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 777,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                    },
                }
            ],
        });

        
        $('#trigger_location_prev_2').on('click', function() {
            $('.location-slider-wrapper-2 .prev-btn').trigger('click');
        });
        $('#trigger_location_next_2').on('click', function() {
            $('.location-slider-wrapper-2 .next-btn').trigger('click');
        });

       

        /*
       Related Portfolio Slider
       ============================*/
        $("#related_portfolio").slick({
            slidesToShow: 3,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='portfolio-arrow-btn prev-btn'><i class='fa-solid fa-angle-left'></i></button>",
            nextArrow:
                "<button type='button' class='portfolio-arrow-btn next-btn'><i class='fa-solid fa-angle-right'></i></button>",
            responsive: [
                {
                    breakpoint: 767,
                    settings: {
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 460,
                    settings: {
                        slidesToShow: 1,
                    },
                },
            ],
        });


        /*
        Process Slider
        ===========================*/
        //  // Process Step Slider
        $(".process-step-slider").slick({
            slidesToShow: 1,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: false,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 800,
            asNavFor: '.process-slider-tab',
            prevArrow:
                "<button type='button' class='process-arrow-btn prev-btn'><i class='icon-arrow-left-2'></i></button>",
            nextArrow:
                "<button type='button' class='process-arrow-btn next-btn active'><i class='icon-arrow-right-2'></i></button>",
                responsive: [
                {
                    breakpoint: 767,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                        arrows: false,
                    },
                },
            ],
        });

        // // Process Step Slider Tab
        $('.process-slider-tab').slick({
            slidesToShow: 4,
            slidesToScroll: 4,
            loop: true,
            infinite: true,
            autoplay: true,
            asNavFor: '.process-step-slider',
            dots: false,
            centerMode: true,
            focusOnSelect: true,
            arrows:false,
            // responsive: [
            //     {
            //         breakpoint: 768,
            //         settings: {
            //             slidesToShow: 3,
            //         },
            //     },
            //     {
            //         breakpoint: 768,
            //         settings: {
            //             slidesToShow: 3,
            //         },
            //     },
            // ],
        });


         // Process Step Slider
         $("#testimonial_one").slick({
            slidesToShow: 1,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1600,
            asNavFor: '.testimonial-user-wrapper',
            prevArrow:
                "<button type='button' class='process-arrow-btn prev-btn'><i class='icon-arrow-left-2'></i></button>",
            nextArrow:
                "<button type='button' class='process-arrow-btn next-btn active'><i class='icon-arrow-right-2'></i></button>",
                responsive: [
                {
                    breakpoint: 767,
                    settings: {
                        autoplay: true,
                        slidesToShow: 1,
                        arrows: false,
                    },
                },
            ],
        });

        // Process Step Slider Tab
        $('.testimonial-user-wrapper').slick({
            slidesToShow: 2,
            loop: false,
            infinite: false,
            asNavFor: '#testimonial_one',
            dots: false,
            centerMode: true,
            focusOnSelect: true,
            arrows:false,
            draggable: true,
            autoplay: true,
        });  

        $('#user-slider-arrow').on('click', function() {
            $('#testimonial_one .next-btn').trigger('click');
        });     

        /*
       Post Gallery Slider
       ============================*/
        $(".post-gallery").slick({
            slidesToShow: 1,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: true,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 300,
            rtl: false,
            prevArrow:
                "<button type='button' class='post-gallery-btn prev-btn'><i class='fa fa-arrow-left'></i></button>",
            nextArrow:
                "<button type='button' class='post-gallery-btn next-btn'><i class='fa fa-arrow-right'></i></button>",
        });
        /*
       Client Logo Slider
       ============================*/
        $(".client-logo-slider-wrapper").slick({
            slidesToShow: 5,
            infinite: true,
            autoplay: true,
            draggable: true,
            arrows: false,
            slidesToScroll: 1,
            loop: true,
            dots: false,
            speed: 1500,
            rtl: false,
            prevArrow:
                "<button type='button' class='post-gallery-btn prev-btn'><i class='fa fa-arrow-left'></i></button>",
            nextArrow:
                "<button type='button' class='post-gallery-btn next-btn'><i class='fa fa-arrow-right'></i></button>",
            responsive: [
                {
                    breakpoint: 1200,
                    settings: {
                        slidesToShow: 4,
                    },
                },
                {
                    breakpoint: 992,
                    settings: {
                        slidesToShow: 3,
                    },
                },
                {
                    breakpoint: 991,
                    settings: {
                        slidesToShow: 3,
                    },
                },
                {
                    breakpoint: 767,
                    settings: {
                        slidesToShow: 2,
                    },
                },
                {
                    breakpoint: 460,
                    settings: {
                        slidesToShow: 1,
                    },
                },
            ],
        });

         /*
        Isotope Grid Js
        ============================*/
        $('.portfolio-filter').on('click', 'li', function() {
            $("li").removeClass("active");
            $(this).addClass("active");
            var filterValue = $(this).attr('data-filter');
            $grid.isotope({
                filter: filterValue
            });
        });

        var $grid = $('.isotope-grid').isotope({
            itemSelector: '.masonry-portfolio-item',
            percentPosition: true,
            masonry: {
                columnWidth: '.masonry-portfolio-item',
                columnHeight: '.masonry-portfolio-item',
                rowWidth: '.masonry-portfolio-item',
                rowHeight: '.masonry-portfolio-item',
                
            }
        });

        $grid.imagesLoaded().progress( function() {
            $grid.isotope('layout');
        });


       

         /*
       Jquery Header Search
       ============================*/
        $('.search-btn').on('click', function (e) {
            e.preventDefault();
            $('body').css('overflow', 'hidden');

            $('.search-form-wrapper').addClass('active');
        });
        $('.search-close').on('click', function (e) {
            e.preventDefault();
            $('body').css('overflow', 'auto');
            $('.search-form-wrapper').removeClass('active');
        });

        window.onclick = function(e){
            if( e.target.matches(".search-form-wrapper") ){
                $('.search-form-wrapper').removeClass('active');
                $('body').css('overflow', 'auto');
            }
        }


        /*
        Counter Js
        ============================*/
        $(".counter").counterUp({
            delay: 10,
            time: 1000,
        });

        /*
       Magnific Popup
       ============================*/
        $(".video-play").magnificPopup({
            disableOn: 700,
            type: "iframe",
            mainClass: "mfp-fade",
            removalDelay: 160,
            preloader: false,
            fixedContentPos: false,
        });

        /*
        Jquery Wow Js
        ============================*/

	if ($('.wow').length) {
		var wow = new WOW(
			{
				boxClass: 'wow',      // animated element css class (default is wow)
				animateClass: 'animated', // animation css class (default is animated)
				offset: 0,          // distance to the element when triggering the animation (default is 0)
				mobile: false,       // trigger animations on mobile devices (default is true)
				live: true       // act on asynchronously loaded content (default is true)
			}
		);
		wow.init();
	}

        /*
       Jquery Nice Select Js
       ============================*/
        $('select.select_option, select.wpcf7-select').niceSelect();

        /*
       Jquery Tilt Js
       ============================*/
        $('.tilt-animate').tilt({
            maxTilt: 12,
            perspective: 1500,
        })


       

                // Initialize Select2 on the select element
        $('#state').select2();

        // Click event handler for the label
        $('.select-location').click(function() {
            // Trigger the Select2 dropdown when the label is clicked
            $('#state').select2('open');
        });

        $(".select2-search__field").attr("placeholder", "Your Placeholder Text");

        // Date Picker
        $("#datepicker").datepicker({
            dateFormat: "dd M, D", // Date format: "26 Mar, Fri"
            onSelect: function(dateText, inst) {
            // $("#datepicker").text("" + dateText);
            }
        });
        $("#datepicker2").datepicker({
            dateFormat: "dd M, D", // Date format: "26 Mar, Fri"
            onSelect: function(dateText, inst) {
            // $("#datepicker").text("" + dateText);
            }
        });


         /*
        Scroll To Top Js
        ============================*/
        $(function () {
            $("#scrollTop").hide();
            var position = $(window).scrollTop();
            var timer;
            $(window).on('scroll', function () {
                var scrollTop = $(window).scrollTop();
                clearTimeout(timer);
                if (scrollTop > 100) {
                    if (scrollTop > position) {
                        $('#scrollTop').fadeOut();
                    } else {
                        $('#scrollTop').fadeIn();
                        timer = window.setTimeout(function() {
                            $("#scrollTop").fadeOut();
                        }, 3000);
                    }
                    position = scrollTop;
                } else {
                    $('#scrollTop').fadeOut();
                }
            });
            $(".scrollup-btn").click(function() {
                $("html, body").animate({ scrollTop: 0 }, "slow");
                return false;
            });
        });

        
        $(function () {
            //Price Range
            $(document).on("click", function (e) {
                var priceRangeInner = $(".price-range-inner");
                
                // Check if the click event target is not #priceLabel, #priceValue, or .price-range-inner
                if (!$(e.target).is(".priceLabel, #priceValue, .price-range-inner")) {
                    priceRangeInner.removeClass("active");
                }
            });

            $(".priceLabel, #priceValue").on("click", function (e) {
                var priceRangeInner = $(".price-range-inner");
                
                if (priceRangeInner.hasClass("active")) {
                    priceRangeInner.removeClass("active");
                } else {
                    priceRangeInner.addClass("active");
                }

                // Prevent the click event from propagating to the document
                e.stopPropagation();
            });
            var minSlider = document.getElementById('min');
            var maxSlider = document.getElementById('max');

            var outputMin = document.getElementById('min-value');
            var outputMax = document.getElementById('max-value');

            outputMin.innerHTML = minSlider.value;
            outputMax.innerHTML = maxSlider.value;

            minSlider.oninput = function(){
            outputMin.innerHTML=this.value;    
            }

            maxSlider.oninput = function(){
                outputMax.innerHTML=this.value; 
            }
        });

        /*
        Preeloader
        ============================*/
        $(window).on("load", function () {
            $("#preloader").fadeOut();
            $("#preloader-status").delay(200).fadeOut("slow");
            $("body").delay(200).css({"overflow-x": "hidden"});
        });

    });
})(jQuery);
